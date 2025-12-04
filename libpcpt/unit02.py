"""
Module: libpcpt.unit02
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

# ================================================
# Exercise 1: exercise_class_rectangle()
# ================================================
def exercise_class_rectangle():
    """Exercise 1: Simple Class

    Notebook: PCPT_02_classes.ipynb
    """
    class Rectangle:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def area(self):
            """Return the area of the rectangle."""
            return self.width * self.height

        def diagonal(self):
            """Return the length of the diagonal using the Pythagorean theorem."""
            return (self.width ** 2 + self.height ** 2) ** 0.5

        def is_square(self):
            """Return True if the rectangle is a square, else False."""
            return self.width == self.height

    # Test examples
    r1 = Rectangle(4, 5)
    print(f"Input: r1 = Rectangle(4, 5)")  
    print(f"r1.area(): {r1.area()}") 
    print(f"r1.diagonal(): {r1.diagonal()}")
    print(f"r1.is_square(): {r1.is_square()}")

    print() 

    r2 = Rectangle(3, 3)
    print(f"Input: r2 = Rectangle(3, 3)")  
    print(f"r2.area(): {r2.area()}") 
    print(f"r2.diagonal(): {r2.diagonal()}")
    print(f"r2.is_square(): {r2.is_square()}")


# ================================================
# Exercise 2: exercise_attributes()
# ================================================
def exercise_attributes():
    """Exercise 2: Class vs Instance Attributes

    Notebook: PCPT_02_classes.ipynb
    """
    class Counter:
        count = 0  # Class attribute shared across all instances

        def __init__(self, name):
            self.name = name      # Instance attribute unique to each instance
            Counter.count += 1
            print(f"Instance name: {self.name}, Counter.count = {Counter.count}")

    # Create instances
    print('Creating: c1 = Counter("Alpha")')
    c1 = Counter("Alpha")

    print('Creating: c2 = Counter("Beta")')
    c2 = Counter("Beta")

    print('Creating: c3 = Counter("Gamma")')
    c3 = Counter("Gamma")

    print()
    print("You can access the class attribute from any instance or directly via the class.")
    print(f"Access via instance (c1.count): {c1.count}")
    print(f"Access via class (Counter.count): {Counter.count}")
 
# ================================================
# Exercise 3: exercise_inheritance_shape()
# ================================================ 
def exercise_inheritance_shape():
    """Exercise 3: Using Inheritance

    Notebook: PCPT_02_classes.ipynb
    """
    class Shape:
        def __init__(self, name):
            self.name = name

        def describe(self):
            print(f"This is a shape called '{self.name}'.")

    class Circle(Shape):
        def __init__(self, name, radius):
            super().__init__(name)
            self.radius = radius

        def describe(self):
            super().describe()
            area = 3.14159 * self.radius ** 2
            print(f"It is a circle with radius {self.radius} and area {area}.")

    # Test the class
    print('Input: c = Circle("MyCircle", 2)')
    c = Circle("MyCircle", 2)
    print("\nOutput of c.describe():")
    c.describe()
