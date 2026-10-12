"""Per-client audio resources, recording, Opus encoding and downlink pacing."""
import asyncio
from pathlib import Path
from bridge.transport.protocols.audio import PcmPacketFormat, OpusPacketFormat
from bridge.audio.codec import LibOpusEncoder
from bridge.audio.recording import RollingWavRecorder
from bridge.transport.protocols.conversation import Kind
from bridge.diagnostics.logging import event
from bridge.audio.pacing import AudioPacer

class AudioPipeline:
    def __init__(self, config):
        self.config = config
        self.pcm_packet_format = PcmPacketFormat(config)
        self.opus_packet_format = OpusPacketFormat(config)
        self.opus_encoder = LibOpusEncoder(config)
        self.output_pcm_queue = asyncio.Queue(maxsize=128)
        self.output_gate = asyncio.Event(); self.output_gate.set()
        self.input_recorder = RollingWavRecorder(config.input_sample_rate, config.channels,
            config.sample_width, Path(config.recordings_dir))
        self.output_recorder = RollingWavRecorder(config.output_sample_rate, config.channels,
            config.sample_width, Path(config.recordings_dir))
        self.inbound_started = False
        self.output_active = self.output_finished = False
        self.output_sequence = self.output_pts_samples = 0
        self.output_pending_pcm = bytearray()
        self.output_pacer = self.new_pacer()

    def new_pacer(self):
        return AudioPacer(self.config.output_send_interval_sec,self.config.output_pacing_max_lag_sec)

    def reset_output(self):
        if hasattr(self,'downlink'): self.downlink.reset()
        self.output_pacer=self.new_pacer()
        self.output_sequence = self.output_pts_samples = 0
        self.output_pending_pcm.clear()
        self.output_active = self.output_finished = False
        self.output_recorder.close()
        while not self.output_pcm_queue.empty(): self.output_pcm_queue.get_nowait()

    async def _accept_audio(self, packet):
        if self.conversation.identity is None: return False
        if not self.inbound_started:
            self.inbound_started = True
            event('INPUT_FIRST_APPEND', self.conversation.identity, seq=packet['seq'])
            if self.config.save_input_wav:
                self.input_recorder.start('uplink', f'_{self.conversation.identity.turn}')
        if self.config.save_input_wav: self.input_recorder.write(packet['payload'])
        return await self.xai.send_audio_append(packet['payload'])

    async def _udp_output_sender(self):
        while True:
            token,chunk=await self.output_pcm_queue.get()
            if not self.output_valid(token):
                self.counters["stale_output"]+=1;continue
            await self.output_gate.wait()
            if not self.output_valid(token): continue
            if chunk is None:
                await self._flush_tagged(token,True)
                if not self.output_valid(token): continue
                await self.output_pacer.wait()
                if not self.output_valid(token): continue
                inner=self.opus_packet_format.pack_end_packet(self.output_sequence,self.output_pts_samples)
                if hasattr(self,'downlink'):
                    self.downlink.finish(token,self.output_sequence,inner)
                else:
                    for i in range(self.config.output_end_repeat_count):
                        if not self.output_valid(token): break
                        self._send(Kind.OPUS,token,self.output_sequence,inner)
                        if i+1<self.config.output_end_repeat_count: await asyncio.sleep(self.config.output_end_repeat_interval)
                if not self.output_valid(token): continue
                self.output_active=False;self.output_finished=True;self.output_recorder.close()
                event("OUTPUT_END",token,final_seq=self.output_sequence,pts=self.output_pts_samples)
                event("OUTPUT_PACING",token,**self.output_pacer.stats())

                # Keep identity for CANCEL while CORE still drains its jitter/DMA.
                continue
            if not self.output_active:
                self.output_active=True
                event("OUTPUT_FIRST_PCM",token,bytes=len(chunk))
                if self.config.save_output_wav:
                    self.output_recorder.start('downlink',f'_{token.turn}')
            if self.config.save_output_wav: self.output_recorder.write(chunk)
            self.output_pending_pcm.extend(chunk)
            await self._flush_tagged(token,False)

    async def _flush_tagged(self,token,final):
        pacer=self.output_pacer
        while self.output_valid(token) and (len(self.output_pending_pcm)>=self.config.opus_frame_bytes or (final and self.output_pending_pcm)):
            frame=bytes(self.output_pending_pcm[:self.config.opus_frame_bytes])
            del self.output_pending_pcm[:self.config.opus_frame_bytes]
            frame+=b'\0'*(self.config.opus_frame_bytes-len(frame))
            payload=self.opus_encoder.encode_pcm16_frame(frame)
            inner=self.opus_packet_format.pack_audio_packet(self.output_sequence,self.output_pts_samples,self.config.opus_frame_samples,payload)
            await pacer.wait()
            # Cancellation/reset can happen while sleeping. Never send old audio.
            if not self.output_valid(token) or pacer is not self.output_pacer: return
            if hasattr(self,'downlink'):
                self.downlink.remember(token,self.output_sequence,inner)
            self._send(Kind.OPUS,token,self.output_sequence,inner)
            pacer.sent()
            self.counters["opus_frames"]+=1
            self.output_sequence+=1;self.output_pts_samples+=self.config.opus_frame_samples

