from flexx import flx, ui
from gui_window import CamvizWindow


class FeatureExplorerComponent(CamvizWindow):

    def init(self):
        ui.Widget(
            title="Feature Explorer",
            style='background-color: #eefafe;'
        )
        # TODO

    @flx.action
    def reset(self):
        # TODO
        pass
