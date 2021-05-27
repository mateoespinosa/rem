"""
Rule set loader class abstracts a widget capable of loading a
Ruleset object from a serialized file.
"""

from flexx import flx, ui, app
from dnn_rem.rules.rule import Rule
from gui_window import CamvizWindow
from dnn_rem.rules.ruleset import Ruleset
from uploader import FileUploader


class RulesetUploader(CamvizWindow):
    """
    Simple flexx widget to allow the upload and deserialization of a Ruleset
    object from a path given by the user.
    """

    text = flx.StringProp("Upload Ruleset", settable=True)
    css_class = flx.StringProp("upload-button", settable=True)

    def init(self):
        self.ruleset = Ruleset()
        self.upload_ruleset = FileUploader(
            text=self.text,
            binary=True,
            css_class=self.css_class,
            flex=1,
        )

    #####################################################
    ## Ruleset Loading Methods
    #####################################################

    @flx.reaction('upload_ruleset.load_started')
    def _ruleset_load_started(self, event):
        return self.ruleset_load_started(event)

    @flx.emitter
    def ruleset_load_started(self, event):
        return event

    @flx.emitter
    def ruleset_load_ended(self):
        return {'ruleset': self.ruleset}

    @flx.reaction('upload_ruleset.file_loaded')
    def _open_data_path(self, *events):
        data_bin_str = events[-1]['filedata']
        try:
            self.ruleset.from_binary_blob(data_bin_str)
            self.ruleset_load_ended()
        except Exception as e:
            self.loading_error({
                'exception': e,
            })

    @flx.emitter
    def loading_error(self, event):
        return event


if __name__ == '__main__':
    """
    Testing code for this widet.
    """

    class MyApp(flx.PyComponent):
        def init(self):
            with flx.VBox():
                self.uploader = RulesetUploader(text="Try me")
                self.state = flx.Label(text="unloaded")

        @flx.reaction('uploader.ruleset_load_started')
        def _load_start(self, *events):
            self.state.set_text('loading started...')

        @flx.reaction('uploader.loading_error')
        def _load_error(self, *events):
            for event in events:
                self.state.set_text(f'Error: {event["exception"]}')

        @flx.reaction('uploader.ruleset_load_ended')
        def _load_ended(self, *events):
            self.state.set_text('loading ended...')
            for event in events:
                print("Finish up loading:")
                print(event["ruleset"])

    my_app = flx.App(MyApp)
    my_app.launch('browser')
    flx.run()
