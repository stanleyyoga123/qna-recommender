class Cleaner:
    def __init__(self):
        pass

    def clean(self, data):
        data['eng'] = data['eng'].apply(lambda x: x.lower())
        return data