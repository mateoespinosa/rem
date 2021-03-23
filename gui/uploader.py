from flexx import flx, app


class FileUploader(flx.BaseButton):
    CSSS = """

    .flx-FileUploader {
        background: #e8e8e8;
        border: 1px solid #ccc;
        transition: background 0.3s;
        padding: 0;
    }
    .flx-FileUploader:hover {
        background: #e8eaff;
    }
    """

    DEFAULT_MIN_SIZE = 10, 28
    file_name = flx.StringProp('', settable=True)

    def _render_dom(self):
        global document, FileReader
        self.action_but = document.createElement('input')
        self.file = document.createElement('input')

        self.action_but.type = 'button'
        self.action_but.style = "width: 100%; height: 100%; display: block;"
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
        return [self.action_but, self.file]

    def __start_loading(self, *events):
        print("Reacting to click!")
        self.file.click()

    @flx.reaction('disabled')
    def __disabled_changed(self, *events):
        if events[-1].new_value:
            self.action_but.setAttribute("disabled", "disabled")
        else:
            self.action_but.removeAttribute("disabled")

    def _handle_file(self):
        print("In handle file with", self.file.files.length, "options")
        if self.file.files.length > 0:
            self.set_file_name(self.file.files[0].name)
            print("Selected filename")
            self.file_selected()
            print("Reading it now")
            self.reader.readAsText(self.file.files[0])

    @flx.emitter
    def file_loaded(self, event):
        return {
            'filedata': event.target.result,
            'filename': self.file_name,
        }

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
