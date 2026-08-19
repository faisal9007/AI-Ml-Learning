

"""def calculate_final_price(price):
    tax_rate = 0.25
    final_price = price + (price * tax_rate)
    return final_price
calculate_final_price(3000) #shirt
calculate_final_price(5000) #pants
calculate_final_price(2000) #shoes
calculate_final_price(1000) #hat

final_price_shirt = calculate_final_price(3000)
final_price_pants = calculate_final_price(5000)
final_price_shoes = calculate_final_price(2000)
final_price_hat = calculate_final_price(1000)

final_prices = {
    "shirt": final_price_shirt,
    "pants": final_price_pants,
    "shoes": final_price_shoes,
    "hat": final_price_hat
}
print(final_prices)

def greet (name):
    return f"Hello, {name}! Welcome to our store."

print(greet("Alice"))
print(greet("Bob"))
print(greet("Charlie"))

def say_hi():
    print("Hi there! How can I assist you today?")
say_hi()
"""
"""name = "Sadia"
print(f"Hello, {name}! Welcome to our store.")
name = "faisal"
print(f"Hello, {name}! Welcome to my programming journey.")
print(f"Hello, {name}! How are you?")
"""
"""def greet (name):
    print(f"Hello, {name}! Welcome to our store.")
    print(f"Hello, {name}! How are you?")
    return 
greet("faisal ")
greet("Sadia ")

a = 12
b = 3
c = 0
print(f"The value of a is: {a}")
print(bool(a))  # Output: True
print(f"The value of b is: {b}")
print(bool(b))  # Output: True
print(f"The value of c is: {c}")
print(bool(c))  # Output: False"""

#print(3+4*2)
#print(15//4+15%4)
#print(3+2**2*5-1)

# range (10,25,4)
# range (10,55,4)
# range (51,255)

# # for i in range (10,55,7):
# #     print(i)
# for i in range (5,56,5):
#     print(i)

# class Dog:
#     def __init__(self, name):
#         print("init is running!")
#         self.name = name

# d = Dog("Rex")
# from unicodedata import name


# def prepare_coffee(coffee_beans,milk, sugar, name="Customer"):
#     name= input("Enter your name: ")
#     print(f"Hello Mr. {name}, preparing your coffee with {coffee_beans}g beans, {milk}g milk, and {sugar}g sugar.")

# prepare_coffee(25,30,10)

# x=5
# print(type(x))
# x=5.5
# print(type(x))
# x=8.9
# print(type(int(x)))


# age = 80
# max_age = 120
# print(age/max_age)

# salary = 150000
# max_salary = 200000
# print(salary/max_salary)

# def normalize_data (data, max_value):
#     normalized = data / max_value
#     print(f'Original {data}:Normalized {normalized}')

#     normalize_data(80, 120)
#     normalize_data(150000, 200000)

# age = int(input("Enter your age: "))
# print(f"Next year, your age will be: {age + 1}")

# price = float(input("Enter the price of the item: "))
# print(f"The price with tax is: {price * 1.15:.3f}")

# def send_email(recipient, subject, body):
#     print(f"user_mail: {recipient}")
#     print(f"Subject: {subject}")
#     print(f"Body: {body}")

# send_email("ofaisal9007@gmail.com", "Welcome!", "Thank you for signing up for our service.")

# send_email("motionwithfaisal@gmail.com", "Have a great day!", "Let's Party Next weekend!")

# send_email(recipient='Your OTP is 1234"',
#            subject="Don't share your OTP with anyone",
#            body="Please use this OTP to verify your account.")
# def calculate_grade_print(marks):
#     grade = "A" if marks >= 80 else "B" # logic
#     print("Grade:", grade)

# def calculate_grade_return(marks):
#     grade = "A" if marks >= 80 else "B" # logic
#     return grade


# #Output is shown, but cannot be reused
# result = calculate_grade_print(85)
# print("Stored result:", result)
# result = calculate_grade_return(85)
# print("Stored result:", result)
# print("Grade:", result)


# # -------------------------------
# # Using returned value
# # -------------------------------

# student_grade = calculate_grade_return(85)
# print("Grade:", student_grade)

# result = calculate_grade_return(85)
# print("Stored result:", result)
# print("Grade:", result)

# def check_scolarship(grade):
#   if grade == "A":
#       print("Eligible for scholarship")
#   else:
#       print("Keep improving!")


# result = calculate_grade_return(85)
# print("Stored result:", result)
# print("Grade:", result)
# check_scolarship(result)

# n = int(input("Enter a number: "))

# for i in range(n, 0, -1):
#     print(i)

# # ছোট ওয়ান-লাইনার ফাংশন
# normalize = lambda x, min_v, max_v: (x - min_v) / (max_v - min_v)
# print(normalize(25, 0, 100)) # Output: 0.25

# model_params = {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}

# def train_model(**kwargs):
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")

# train_model(**model_params)

#task - 1
# name = 'Faisal'
# age = 36
# country = 'Saudi Arabia'
# learning_ai = 'Al & Ml Engineering'

# print (name)
# print (age)
# print(f"Hello, {name}! How are you? You're learning {learning_ai}, You're Living in {country}, Your {age} Years old")

# #task - 2

# monthly_salary = 5000
# yearly_salary = monthly_salary*12
# new_salary = monthly_salary + (monthly_salary*10/100)

# print (new_salary)

# delivery_time = 35
# distance = 8
# order_value = 75

# import pandas as pd
# import numpy as np

# try:
#     from sklearn.linear_model import LinearRegression  # type: ignore[import-not-found]
# except ImportError:
#     class linear_model:
#         """A lightweight fallback implementation of a linear regression model."""

#         def __init__(self):
#             self.intercept_ = 0.0
#             self.coef_ = None

#         def fit(self, X, y):
#             X = np.asarray(X, dtype=float)
#             y = np.asarray(y, dtype=float)

#             if X.ndim == 1:
#                 X = X.reshape(-1, 1)

#             if X.shape[0] != y.shape[0]:
#                 raise ValueError("X and y must contain the same number of samples.")

#             X_with_bias = np.c_[np.ones(X.shape[0]), X]
#             coefficients, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)

#             self.intercept_ = float(coefficients[0])
#             self.coef_ = coefficients[1:]
#             return self

#         def predict(self, X):
#             X = np.asarray(X, dtype=float)
#             if X.ndim == 1:
#                 X = X.reshape(-1, 1)
#             if self.coef_ is None:
#                 raise ValueError("Model must be fitted before calling predict().")
#             return self.intercept_ + X @ self.coef_

#         def score(self, X, y):
#             y = np.asarray(y, dtype=float)
#             predictions = self.predict(X)
#             ss_res = np.sum((y - predictions) ** 2)
#             ss_tot = np.sum((y - np.mean(y)) ** 2)
#             if ss_tot == 0:
#                 return 1.0
#             return 1.0 - (ss_res / ss_tot)

#     LinearRegression = linear_model

# # 1. Prepare your data (Features = X, Target = y)
# data = {
#     'distance': [8],
#     'order_value': [75],
#     'delivery_time': [35]  # This is what we want to predict
# }

# df = pd.DataFrame(data)

# X = df[['distance', 'order_value']]  # Features
# y = df['delivery_time']              # Target

# if LinearRegression is not None:
#     # 2. Train the Machine Learning model
#     model = LinearRegression()
#     model.fit(X, y)

#     # 3. Predict delivery time for a new order
#     new_order = [[8, 75]]  # 8 miles distance, $75 order value
#     predicted_time = model.predict(new_order)
#     print(f"Predicted Delivery Time: {predicted_time[0]:.1f} minutes")
# else:
#     print("scikit-learn is not installed. Install it with: pip install scikit-learn")

technologies = ["Python", "JavaScript", "Java", "C++", "C#", "Swift", "Go", "Rust"]
print("List of Technologies:")
print(0, technologies[0])
technologies.append("pytorch")
print(len(technologies))
print(1, technologies[8])

#task 2


# person = {
#     "name": "Alice",
#     "age": 30,
#     "country": "USA",
#     "profession": "Engineer",
#     "learning": "Python"
# }

# print("Person Information:")
# print("Name:", person["name"])
# print("Age:", person["age"])
# print("Country:", person["country"])    
# print("Profession:", person["profession"])
# print("Learning:", person["learning"])
 #task 3
# order = {
#    "distance": "10 km",
#    "restaurant_rating": 4.3,
#    "order_value": "90 SAR",
#    "number_of_items": 5,
#    "delivery_time": "40 minutes"
# }

# print("Order Information:")
# order['number_of_orders'] = 15
# print("Distance:", order["distance"])   
# print("Restaurant Rating:", order["restaurant_rating"])
# print("Order Value:", order["order_value"])
# print("Number of Items:", order["number_of_items"])
# print("Delivery Time:", order["delivery_time"])
# print("Number of Orders:", order["number_of_orders"])

# orders = [
#     {
#         "distance": 7,
#         "restaurant_rating": 4.4,
#         "number_of_items": 4,
#         "delivery_time": 30
#     },
#     {
#         "distance": 9,
#         "restaurant_rating": 4.8,
#         "number_of_items": 8,
#         "delivery_time": 27
#     },
#     {
#         "distance": 12,
#         "restaurant_rating": 4.4,
#         "number_of_items": 6,
#         "delivery_time": 24
#     }
# ]
# print("Orders Information:")
# print("Order 2 Delivery Time:", orders[1]["delivery_time"])

# delivery_time =35

# if delivery_time <= 20:
#     print("Delivery is Fast.")   
# elif delivery_time <= 30:
#     print("Delivery is Normal.")
# else:
#     print("Delivery is Late.")

age = 36

if age < 18:
    print("You are a minor.")
else:
    print("You are an adult.")

delivery_time = 42

if delivery_time <= 20:
    print("Delivery is Fast.")
elif delivery_time <= 30:
    print("Delivery is Normal.")
else:
    print("Delivery is Late.")

distance = 12
number_of_items = 6

if distance >= 10 and number_of_items >= 5:
    print("Eligible for Free Delivery.")
else:
    print("Not Eligible for Free Delivery.")

if distance <= 10 and number_of_items <= 5:
    print("High Delivery Priority.")
else:
    print("Normal Delivery Priority.")

order = {
    "distance": 12,
    "number_of_items": 6,
    "delivery_time": 42
}

if order["distance"] >= 10 and order["number_of_items"] >= 5:
    print("Eligible for Free Delivery.")
else:
    print("Not Eligible for Free Delivery.")

if order["distance"] >= 10 and order["number_of_items"] >= 5:
    print("High Delivery Load.")
else:
    print("Normal Delivery Load.")

if order["delivery_time"] <= 20:
    print("Delivery is Fast.")
elif order["delivery_time"] <= 30:
    print("Delivery is Normal.")
else:
    print("Delivery is Late.")