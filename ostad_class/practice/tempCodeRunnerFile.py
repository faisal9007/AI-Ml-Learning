if order["delivery_time"] <= 20:
    print("Delivery is Fast.")
elif order["delivery_time"] <= 30:
    print("Delivery is Normal.")
else:
    print("Delivery is Late.")