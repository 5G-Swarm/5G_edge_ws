

class _Functor(object):
# """
#     A functor is a named function with its own state.

#     :NOTE: you should use and only use FunctorFactory to create functor objects.
# """
    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
    # """
    #         :NOTE: to be implemented

    #         :param args: 
    #         :param kwargs: 
    #         :return: 
    #         """
        pass


class FunctorFactory(object):
# """
#     Produces functor objects;

#     :NOTE: Inserts the functor objects in the global dictionary so that the name of the functors become available even
#            though they are not explicitly defined.
#     """
    def create(self,name):
        globals()[name] = _Functor(name)


# examples
if __name__ == '__main__':
    
    factory = FunctorFactory()
    for name in ("funca", "funcb", "funcc"):
        functor = _Functor(name)
        factory.create(name)
    print(funca())
    print(funcb())
    print(funcc())