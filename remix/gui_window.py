from flexx import flx


class CamvizWindow(flx.PyComponent):
    def init(self):
        super().init()

    @flx.emitter
    def ruleset_update(self, event):
        return event

    @flx.action
    def reset(self):
        pass

    @flx.action
    def perform_update(self, event):
        # By default, this will trigger a whole reset
        self.reset()
