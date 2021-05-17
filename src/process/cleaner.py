import pandas as pd


MAPPING = {
    '\n': ". ",
    '\\': "\frg",
}


def process_sentences(text):
    new = text.lower()
    for mapping in MAPPING:
        new.replace(mapping, MAPPING[mapping])
    return new


class Cleaner:
    def __init__(self):
        pass

    def clean(self, data):
        data['eng'] = data['eng'].apply(process_sentences)
        return data


# TESTING
p = Cleaner()
a = p.clean(pd.DataFrame([{'eng': ' an \( l-c-r \) sEEGRGRries circuit with \( l= \)\n\..'},
                          {'eng': "defrgthfrgtehnrgr"}]))
print(a)
