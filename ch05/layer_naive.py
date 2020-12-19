class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1

apple_layer = MulLayer()
tax_layer = MulLayer()

apple_price = apple_layer.forward(apple, apple_num)
total_price = tax_layer.forward(apple_price, tax)

print(apple_price)
print(total_price)

dapple_price, dtax = tax_layer.backward(1)
dapple, dapple_num = apple_layer.backward(dapple_price)

print(dapple)
print(dapple_num)
