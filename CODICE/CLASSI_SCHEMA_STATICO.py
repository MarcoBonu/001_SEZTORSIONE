class CONTENITORI_SCHEMA_STATICO(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value