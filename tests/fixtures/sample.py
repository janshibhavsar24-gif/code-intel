"""Sample Python fixture used by extractor and resolve tests."""


class Animal:
    """Base animal class."""

    def speak(self):
        """Make a sound."""
        return make_sound(self)

    def breathe(self):
        return True


class Dog(Animal):
    """A dog."""

    def speak(self):
        return bark()


def make_sound(animal):
    return animal.breathe()


def bark():
    return "woof"


def _private():
    pass
