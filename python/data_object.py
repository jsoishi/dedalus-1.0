def create_data(representation, shape, name):
    new_class = type(name, (BaseData,), {'representation': representation,
                                           'shape': shape})
    return new_class

class BaseData(object):
    def __init__(self, fields, time):
        self.time = time
        self.fields = {}
        for f in fields:
            self.fields[f] = self.representation(self.shape)

    def __getitem__(self, item):
        a = self.fields.get(item, None)
        if a is None:
            raise KeyError
        return a
