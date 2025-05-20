#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Function to transpose a matrix
void transpose(double **matrix, double **transposed, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

// Function to multiply two matrices
void multiply(double **A, double **B, double **result, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i][j] = 0;
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to create an identity matrix
void create_identity_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// Function to perform Gauss-Jordan elimination for matrix inversion
int invert_matrix(double **matrix, double **inverse, int size) {
    create_identity_matrix(inverse, size);

    for (int p = 0; p < size; p++) {
        double pivot = matrix[p][p];
        if (pivot == 0) return 0; // Singular matrix

        for (int j = 0; j < size; j++) {
            matrix[p][j] /= pivot;
            inverse[p][j] /= pivot;
        }

        for (int i = 0; i < size; i++) {
            if (i != p) {
                double factor = matrix[i][p];
                for (int j = 0; j < size; j++) {
                    matrix[i][j] -= factor * matrix[p][j];
                    inverse[i][j] -= factor * inverse[p][j];
                }
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("error\n");
        return 1;
    }

    FILE *train_file = fopen(argv[1], "r");
    FILE *data_file = fopen(argv[2], "r");

    if (!train_file || !data_file) {
        printf("error\n");
        return 1;
    }

    char buffer[6];
    fscanf(train_file, "%5s", buffer);
    if (strcmp(buffer, "train") != 0) {
        printf("error\n");
        return 1;
    }

    int k, n;
    fscanf(train_file, "%d", &k);
    fscanf(train_file, "%d", &n);

    // Allocate memory for matrices
    double **X = malloc(n * sizeof(double *));
    double **Y = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        X[i] = malloc((k + 1) * sizeof(double));
        Y[i] = malloc(sizeof(double));
        X[i][0] = 1.0; // Add column of 1s for w0
        for (int j = 1; j <= k; j++) {
            fscanf(train_file, "%lf", &X[i][j]);
        }
        fscanf(train_file, "%lf", &Y[i][0]);
    }

    fclose(train_file);

    // Transpose matrix X
    double **XT = malloc((k + 1) * sizeof(double *));
    for (int i = 0; i <= k; i++) {
        XT[i] = malloc(n * sizeof(double));
    }
    transpose(X, XT, n, k + 1);

    // Calculate (XT * X)
    double **XT_X = malloc((k + 1) * sizeof(double *));
    for (int i = 0; i <= k; i++) {
        XT_X[i] = malloc((k + 1) * sizeof(double));
    }
    multiply(XT, X, XT_X, k + 1, n, k + 1);

    // Invert (XT * X)
    double **XT_X_inv = malloc((k + 1) * sizeof(double *));
    for (int i = 0; i <= k; i++) {
        XT_X_inv[i] = malloc((k + 1) * sizeof(double));
    }
    if (!invert_matrix(XT_X, XT_X_inv, k + 1)) {
        printf("error\n");
        return 1;
    }

    // Calculate (XT * Y)
    double **XT_Y = malloc((k + 1) * sizeof(double *));
    for (int i = 0; i <= k; i++) {
        XT_Y[i] = malloc(sizeof(double));
    }
    multiply(XT, Y, XT_Y, k + 1, n, 1);

    // Calculate W = (XT_X_inv) * (XT_Y)
    double **W = malloc((k + 1) * sizeof(double *));
    for (int i = 0; i <= k; i++) {
        W[i] = malloc(sizeof(double));
    }
    multiply(XT_X_inv, XT_Y, W, k + 1, k + 1, 1);

    // Read input data for predictions
    fscanf(data_file, "%5s", buffer);
    if (strcmp(buffer, "data") != 0) {
        printf("error\n");
        return 1;
    }

    int m;
    fscanf(data_file, "%d", &k);
    fscanf(data_file, "%d", &m);

    double **X_new = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        X_new[i] = malloc((k + 1) * sizeof(double));
        X_new[i][0] = 1.0; // Add column of 1s for w0
        for (int j = 1; j <= k; j++) {
            fscanf(data_file, "%lf", &X_new[i][j]);
        }
    }

    fclose(data_file);

    // Predict prices for input data
    double **Y_pred = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        Y_pred[i] = malloc(sizeof(double));
    }
    multiply(X_new, W, Y_pred, m, k + 1, 1);

    // Print predicted prices
    for (int i = 0; i < m; i++) {
        printf("%.0f\n", Y_pred[i][0]);
    }

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(X[i]);
        free(Y[i]);
    }
    for (int i = 0; i <= k; i++) {
        free(XT[i]);
        free(XT_X[i]);
        free(XT_X_inv[i]);
        free(XT_Y[i]);
        free(W[i]);
    }
    for (int i = 0; i < m; i++) {
        free(X_new[i]);
        free(Y_pred[i]);
    }
    free(X);
    free(Y);
    free(XT);
    free(XT_X);
    free(XT_X_inv);
    free(XT_Y);
    free(W);
    free(X_new);
    free(Y_pred);

    return 0;
}
