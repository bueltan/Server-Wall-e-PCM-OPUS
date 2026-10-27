"""Provider selection for the continuous UDP v3 coordinator."""

def create_provider(config, queue, valid, events):
    if config.voice_provider == 'xai':
        from bridge.providers.xai.vad import VadXai
        return VadXai(config, queue, valid, events)
    if config.voice_provider == 'gemini':
        from bridge.providers.gemini.live import GeminiLive
        return GeminiLive(config, queue, valid, events)
    raise ValueError('PIUU_VOICE_PROVIDER must be xai or gemini')
