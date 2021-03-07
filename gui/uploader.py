from flexx import flx, app

class FileInput(flx.LineEdit):

    def _create_dom(self):
        global document, FileReader
        node = super()._create_dom()
        self.file = document.createElement('input')
        self.file.type = 'file'
        self.file.style = 'display: none'
        self.file.addEventListener('change', self._handle_file)
        node.appendChild(self.file)
        self.reader = FileReader()
        self.reader.onload = self.file_loaded
        return node


    def _handle_file(self):
        print("self.file.files", self.file.files)
        self.node.value = self.file.files[0].name
        self.file_selected()


    def select_file(self):
        self.file.click()


    def load(self):
        if self.file.files.length > 0:
            self.reader.readAsText(self.file.files[0])


    @flx.emitter
    def file_loaded(self, event):
        return { 'filedata': event.target.result }


    @flx.emitter
    def file_selected(self):
        return { 'filename': self.node.value }



class Uploader(flx.Widget):
    file_name = flx.StringProp('')

    def init(self):
        self.file_input = FileInput()
        self.pick_file = flx.Button(text='...')
        self.do_upload = flx.Button(text='Upload', disabled=True)


    @flx.reaction('file_input.file_selected')
    def handle_file_selected(self, *events):
        self.set_file_to_upload(events[-1]['filename'])


    @flx.reaction('file_input.file_loaded')
    def handle_file_loaded(self, *events):
        self.file_loaded(events[-1]['filedata'])


    @flx.reaction('pick_file.pointer_click')
    def on_pick_file(self, *events):
        self.file_input.select_file()


    @flx.reaction('do_upload.pointer_click')
    def on_do_upload(self, *events):
        self.file_input.load()


    @flx.action
    def set_file_to_upload(self, value):
        self.do_upload._mutate_disabled(value == '')
        self._mutate_file_name(value)


    @flx.emitter
    def file_loaded(self, data):
        return {'filedata': data }



class App(flx.PyComponent):

    def init(self):
        self.uploader = Uploader()


    @flx.reaction('uploader.file_loaded')
    def handle_file_upload(self, *events):
        print(events[-1]['filedata'])


app.serve(App)
app.start()
