# import awesome_print

class AttrDict(object):
    def __init__(self, dct):
        self.dict = dct

    def __repr__(self):
        return repr(self.dict)

    # def __str__(self):
    #     return awesome_print.awesome_print.format(self.dict)

    def __getattr__(self, attr):
        try:
            val = self.dict[attr]
            if isinstance(val, dict):
                val = AttrDict(val)
            return val
        except KeyError:
            raise AttributeError
