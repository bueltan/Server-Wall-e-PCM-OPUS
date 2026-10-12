import struct
import unittest
from dataclasses import replace
from bridge.config import BridgeConfig
from bridge.conversation_protocol import Identity, Packet, Kind, encode, decode
from bridge.downlink_recovery import DownlinkRecovery

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.sent=[];self.cfg=BridgeConfig();self.r=DownlinkRecovery(self.cfg,lambda *p:self.sent.append(p))
        self.token=Identity(1,2,3,4)
    def request(self,base,mask=1,token=None):
        return Packet(Kind.DOWNLINK_NACK,0,token or self.token,base,struct.pack('!IH',base,mask))
    def test_lost_frames_retransmitted_once_with_bounded_work(self):
        for seq in range(8): self.r.remember(self.token,seq,bytes([seq]),0)
        self.assertTrue(self.r.request(self.request(0,255),.3))
        self.r.poll(.32);self.assertEqual([p[2] for p in self.sent],[0,1])
        self.r.request(self.request(0,255),.34)
        for now in (.34,.36,.38): self.r.poll(now)
        self.assertEqual([p[2] for p in self.sent],list(range(8)))
        self.assertEqual(self.r.stats['repeated'],8)
        self.assertTrue(all(p[0]==Kind.OPUS and p[1]==self.token for p in self.sent))
    def test_cache_is_bounded_and_expiry_uses_original_send_time(self):
        for seq in range(130): self.r.remember(self.token,seq,b'opus',0)
        self.assertEqual(len(self.r.frames),128)
        self.r.request(self.request(0),.5)
        self.r.request(self.request(2),3.01);self.r.poll(3.02)
        self.assertFalse(self.sent)
        self.assertEqual(self.r.stats['cache_miss'],1);self.assertEqual(self.r.stats['expired'],1)
    def test_recovery_expiring_while_queued_is_not_sent(self):
        self.r.remember(self.token,0,b'a',0);self.r.request(self.request(0),2.99);self.r.poll(3.01)
        self.assertFalse(self.sent)
    def test_invalid_and_stale_requests_never_send(self):
        self.r.remember(self.token,0,b'a',0)
        for p in (self.request(0,0),self.request(0,256),self.request(0,token=Identity(1,2,3,5)),
                  Packet(Kind.DOWNLINK_NACK,0,self.token,1,struct.pack('!IH',0,1))):
            self.assertFalse(self.r.request(p,.2))
        self.r.poll(.3);self.assertFalse(self.sent)
    def test_lost_end_and_lost_ack_retry_until_matching_ack(self):
        self.r.finish(self.token,10,b'end',0)
        self.r.poll(.1);self.assertEqual(len(self.sent),1)
        self.r.poll(.16);self.assertEqual(len(self.sent),2)
        wrong=Packet(Kind.AUDIO_END_ACK,0,self.token,9)
        self.assertFalse(self.r.acknowledge(wrong))
        ack=Packet(Kind.AUDIO_END_ACK,0,self.token,10)
        self.assertEqual(decode(encode(ack)),ack)
        self.assertTrue(self.r.acknowledge(ack))
        self.r.poll(.4);self.assertEqual(len(self.sent),2)
    def test_end_timeout_is_failure_not_successful_playback(self):
        self.r.finish(self.token,0,b'end',0)
        self.assertFalse(self.r.poll(5));self.assertEqual(self.r.stats['end_acked'],0)
    def test_cancel_removes_pending_audio_and_end(self):
        self.r.remember(self.token,0,b'a',0);self.r.request(self.request(0),.2)
        self.r.finish(self.token,1,b'end',.2);self.r.reset();self.sent.clear();self.r.poll(.4)
        self.assertFalse(self.sent);self.assertFalse(self.r.acknowledge(Packet(Kind.AUDIO_END_ACK,0,self.token,1)))
    def test_configuration_rejects_unbounded_or_invalid_values(self):
        for name,value in [('output_cache_frames',0),('output_cache_frames',513),('output_end_timeout_sec',float('nan'))]:
            with self.assertRaises(ValueError): DownlinkRecovery(replace(self.cfg,**{name:value}),lambda *p:None)

    def test_local_send_failure_is_bounded_and_end_still_retries(self):
        def fail(*args): raise OSError('local UDP failure')
        self.r.send=fail
        self.r.remember(self.token,0,b'a',0);self.r.request(self.request(0),.2)
        self.r.finish(self.token,1,b'end',.2)
        self.assertEqual(len(self.r.pending),1)  # END did not drain extra audio
        self.assertTrue(self.r.poll(.36))
        self.assertEqual(self.r.stats['send_failed'],3)
        self.assertFalse(self.r.pending)
