from diart.models import SegmentationModel

class ModelLoader:
    # Code example
    # https://github.com/juanmc2005/diart/blob/main/src/diart/models.py#L13
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self):
        return self.model

class DiartWrapper(SegmentationModel):
    # Documentation: 
    # https://github.com/juanmc2005/diart#-custom-models
    # Code example:
    # https://github.com/juanmc2005/diart/blob/main/src/diart/models.py#L94
    # https://github.com/juanmc2005/diart/blob/main/src/diart/models.py#L156

    def __init__(self, model_loader, sample_rate=16000, duration=5):
        super(DiartWrapper, self).__init__(model_loader)
        self._sample_rate = sample_rate
        self._duration = duration

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def duration(self) -> float:
        return self._duration  # seconds

    def forward(self, waveform):
        # self.model is created lazily
        return self.model(waveform)
