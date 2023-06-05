#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const double EPSILON = 1e-11;

#define ADDITIONAL_EXERCISE

void showMatrix(const std::vector<std::vector<double>> &A, const std::vector<int> &index)
{
    int n = A.size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            std::cout << std::setw(8) << A[index[i]][j] << "\t";
        std::cout << "\n";
    }
}

void showVector(const std::vector<double> &v, const std::vector<int> &index)
{
    int n = v.size();
    for (int i = 0; i < n; i++)
    {
        std::cout << std::setw(8) << v[index[i]] << "\n";
    }
}

void decomposeLUWithGauss(std::vector<std::vector<double>> &A, std::vector<int> &index)
{
    int n = A.size();

    index.resize(n);
    for (int i = 0; i < n; i++)
    {
        index[i] = i;
    }

    for (int k = 0; k < n; k++)
    {
        if (A[index[k]][k] == 0.0)
        {
            int pivot_row = k;

            for (int i = k + 1; i < n; i++)
            {
                if (std::abs(A[index[i]][k]) > std::abs(A[index[pivot_row]][k]))
                {
                    pivot_row = i;
                }
            }

            std::swap(index[k], index[pivot_row]);
        }

        for (int i = k + 1; i < n; i++)
        {
            double factor = A[index[i]][k] / A[index[k]][k];
            A[index[i]][k] = factor;

            for (int j = k + 1; j < n; j++)
            {
                A[index[i]][j] -= factor * A[index[k]][j];
            }
        }
    }
}

void solveEquation(const std::vector<std::vector<double>> &A, const std::vector<int> &index, std::vector<double> &b)
{
    int n = A.size();

    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
        {
            sum += A[index[i]][j] * b[index[j]];
        }
        b[index[i]] = (b[index[i]] - sum) / 1.0;
    }

    std::cout << "Solution y (Ly = b): " << std::endl;
    showVector(b, index);

    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            sum += A[index[i]][j] * b[index[j]];
        }

        if (A[index[i]][i] == 0)
        {
            std::cout << "Dividing by 0" << std::endl;
            return;
        }

        b[index[i]] = (b[index[i]] - sum) / A[index[i]][i];
    }
}

void max(const std::vector<double> &x_exact, const std::vector<double> &x_numeric)
{
    std::vector<double> temp(x_exact.size());
    double blad = 0.0;

    for (int i = 0; i < x_exact.size(); ++i)
    {
        temp[i] = std::fabs(x_exact[i] - x_numeric[i]);
    }

    blad = temp[0];

    for (int i = 1; i < x_exact.size(); ++i)
    {
        if (temp[i] > blad)
            blad = temp[i];
    }

    std::cout << "Maximum difference (blad) between analytical and numerical solutions: " << blad << "\n";
}

void max_norm(const std::vector<double> &x_exact, const std::vector<double> &x_numeric)
{
    std::vector<double> diff(x_exact.size());

    for (int i = 0; i < x_exact.size(); ++i)
    {
        diff[i] = x_exact[i] - x_numeric[i];
    }

    double norm_diff = 0.0;
    for (double val : diff)
    {
        norm_diff += val * val;
    }
    norm_diff = std::sqrt(norm_diff);

    std::cout << "Norm of difference (norm_diff) between x_exact and x_numeric: " << norm_diff << "\n";
}

int main()
{
#ifndef ADDITIONAL_EXERCISE
    std::vector<std::vector<double>> A = {
        {1, -20, 30, -4},
        {2, -40, -6, 50},
        {9, -180, 11, -12},
        {-16, 15, -140, 13}};
    std::vector<double> b = {35, 104, -366, -354};
#else
    std::vector<std::vector<double>> A = {
        {1.0 + EPSILON, 1.0, 1.0, 1.0},
        {1.0, 1.0 + EPSILON, 1.0, 1.0},
        {1.0, 1.0, 1.0 + EPSILON, 1.0},
        {1.0, 1.0, 1.0, 1.0 + EPSILON}};
    std::vector<double> b = {6.0 + EPSILON, 6.0 + 2.0 * EPSILON, 6.0 + 2.0 * EPSILON, 6.0 + EPSILON};
    std::vector<double> b_exact = {1.0, 2.0, 2.0, 1.0};
#endif
    std::vector<int> P;

    decomposeLUWithGauss(A, P);
    std::cout << "Matrix A after LU Decomposition: " << std::endl;
    showMatrix(A, P);
    solveEquation(A, P, b);
    std::cout << "Solution x (Ux = y): " << std::endl;
    showVector(b, P);

#ifdef ADDITIONAL_EXERCISE
    max(b, b_exact);
    max_norm(b, b_exact);
#endif

    return 0;
}