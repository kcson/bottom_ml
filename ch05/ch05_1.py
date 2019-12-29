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


apple_price = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_prices = mul_apple_layer.forward(apple_price, apple_num)
total_price = mul_tax_layer.forward(apple_prices, tax)
print(total_price)

dprice = 1
dapple_prices, dtax = mul_tax_layer.backward(dprice)
dapple_price, dapple_num = mul_apple_layer.backward(dapple_prices)
print(dapple_price, dapple_num, dtax)


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


apple_price = 100
apple_num = 2
orange_price = 150
orange_num = 3
tax = 1.1

apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
price_add_layer = AddLayer()
tax_mul_layer = MulLayer()

apple_prices = apple_mul_layer.forward(apple_price, apple_num)
orange_prices = orange_mul_layer.forward(orange_price, orange_num)
add_price = price_add_layer.forward(apple_prices, orange_prices)
total_price = tax_mul_layer.forward(add_price, tax)
print(total_price)

dprice = 1
dadd_price, dtax = tax_mul_layer.backward(dprice)
print(dadd_price, dtax)

dapple_prices, dorange_prices = price_add_layer.backward(dadd_price)
print(dapple_prices, dorange_prices)

dapple_price, dapple_num = apple_mul_layer.backward(dapple_prices)
print(dapple_price, dapple_num)
dorange_price, dorange_num = orange_mul_layer.backward(dorange_prices)
print(dorange_price, dorange_num)
