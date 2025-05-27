# HousingEstimate
A C-based program using Python in order to utilize a full Linear Regression pipeline--built from scratch--to predict housing prices. Reads training data from an input file, builds a linear model using matrix operations, and then uses the learned model to estimate prices for new data. 

Demonstrates how machine learning and linear algebra can be implemented without using external libraries. It only uses standard C and Python for:
- Dynamic memory management
- File I/O
- Matrix operations (Multiplication, Transposition, Inversion via Gauss-Jordan elimination)

The program computes the weights for regression using the normal equation, and then uses the calculated weights to predict outcomes for new data points. 
Testing: (Test on a Linux Machine or Remotely Access a Linux Machine to run this project)
1. Compile: gcc -o estimate main.c
2. Run with provided training data and test files(under "data"): ./estimate train.xx.txt, ./estimate data.xx.txt
3. View output: predicted prices for each category based on learned weights. 
