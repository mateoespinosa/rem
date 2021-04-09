from flexx import flx, app

class FileUploader(flx.BaseButton):
    CSS = """
    .flx-FileUploader {
       padding: 0;
       box-sizing: border-box;
    }
    """

    DEFAULT_MIN_SIZE = 10, 28
    file_name = flx.StringProp('', settable=True)
    binary = flx.BoolProp(False, settable=True)

    def _render_dom(self):
        global document, FileReader
        self.action_but = document.createElement(
            'input'
        )
        self.action_but.className = "flx-Button flx-BaseButton"
        self.file = document.createElement('input')

        self.action_but.type = 'button'
        self.action_but.style = (
            "min-width: 10px; max-width: 1e+09px; min-height: 28px;"
            "max-height: 1e+09px;flex-grow: 1; flex-shrink: 1;"
            "margin-left: 0px; margin: 0; box-sizing: border-box; width: 100%;"
        )
        self.action_but.value = self.text
        self._addEventListener(
            self.action_but,
            'click',
            self.__start_loading,
            0,
        )
        self.file.type = 'file'
        self.file.style = 'display: none'

        self.file.addEventListener(
            'change',
            self._handle_file,
        )

        self.reader = FileReader()
        self.reader.onload = self.file_loaded
        self.reader.onloadstart = self.load_started
        self.reader.onloadend = self.load_ended
        self.reader.onerror = self.reading_error

        return [self.action_but, self.file]

    def __start_loading(self, *events):
        self.file.click()

    @flx.reaction('disabled')
    def __disabled_changed(self, *events):
        if events[-1].new_value:
            self.action_but.setAttribute("disabled", "disabled")
        else:
            self.action_but.removeAttribute("disabled")

    def _handle_file(self):
        if self.file.files.length > 0:
            self.set_file_name(self.file.files[0].name)
            self.file_selected()
            if self.binary:
                self.reader.readAsArrayBuffer(self.file.files[0])
            else:
                self.reader.readAsText(self.file.files[0])

    @flx.emitter
    def file_loaded(self, event):
        return {
            'filedata': event.target.result,
            'filename': self.file_name,
        }

    @flx.emitter
    def load_started(self, event):
        return {}

    @flx.emitter
    def load_ended(self, event):
        return {}

    @flx.emitter
    def reading_error(self, event):
        return event

    @flx.emitter
    def file_selected(self):
        return {
            'filename': self.file_name,
        }


if __name__ == '__main__':

    class App(flx.PyComponent):
        def init(self):
            with flx.VBox():
                self.uploader = FileUploader(text="Try me")
                self.text_box = flx.Label("", flex=1)

        @flx.reaction('uploader.file_loaded')
        def handle_file_upload(self, *events):
            self.text_box.set_html(events[-1]['filedata'])

    app.serve(App)
    app.start()
