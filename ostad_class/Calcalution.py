num1_input = input("Enter the first number: ")
num2_input = input("Enter the second number: ")
num1 = float(num1_input)
num2 = float(num2_input)
# Addition
addition = num1 + num2
 
# Subtraction
subtraction = num1 - num2
 
# Multiplication
multiplication = num1 * num2
 
# Division (handle the case where the second number is zero)
if num2 != 0:
    division = num1 / num2
else:
    division = "Undefined (cannot divide by zero)"

# --- Clearly labeled outputs ---
print("\n----- CALCULATOR RESULTS -----")
print(f"Addition of {num1} and {num2}       = {addition}")
print(f"Subtraction of {num1} and {num2}    = {subtraction}")
print(f"Multiplication of {num1} and {num2} = {multiplication}")
print(f"Division of {num1} and {num2}       = {division}")
print("-------------------------------")