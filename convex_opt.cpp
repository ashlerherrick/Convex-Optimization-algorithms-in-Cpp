#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>

//Prints the vector by iterating through the elements with a for loop.
void print_vector(std::vector<long double> v)
{
    std::cout << '[';
    for(int i = 0;i < v.size();i++)
    {
        std::cout << v.at(i);
        if(i != v.size()-1)
        {
            std::cout << ',';
        }
    }
    std::cout << ']' <<std::endl;
    
}
//Prints a 2 dimensional vector as if it were a matrix
void print_matrix(std::vector<std::vector<long double>> M)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    for(int i = 0; i < M.size(); i++)
    {
        std::cout << '|';
        for(int j = 0; j < M.at(0).size(); j++)
        {
            std::cout << M.at(i).at(j) << ' ';
        }
    std::cout << '|' <<std::endl;     
    }
}

//Function for calculating the lp norm. 
long double lpnorm(std::vector<long double> x, double p)
{
    long double sum = 0;
    for(int i = 0; i < x.size(); i++)
    {
        sum += pow(x[i], p);
    }
    return pow(sum, 1/p);
}

//Calculates the dot product of x and y.
long double dot(std::vector<long double> x, std::vector<long double> y)
{
    long double a = 0;
    for(int i = 0; i < x.size(); i++ )
    {
        a += x[i]*y[i];
    }
    return a;
}

//Performs scalar multiplication on a vector
std::vector<long double> scal_mult(long double lambda, std::vector<long double> x)
{
    for(int i = 0; i < x.size(); i++)
    {
        x[i] *= lambda;
    }
    return x;
}
//Function for elementwise vector addition
std::vector<long double> vec_add(std::vector<long double> x, std::vector<long double> y)
{
    std::vector<long double> z;
    for(int i = 0; i < x.size(); i++)
    {
        z.push_back(x[i] + y[i]);
    }
    return z;
}
//Function for elementwise vector multiplication
std::vector<long double> vec_mult(std::vector<long double> x, std::vector<long double> y)
{
    std::vector<long double> z;
    for(int i = 0; i < x.size(); i++)
    {
        z.push_back(x[i] * y[i]);
    }
    return z;
}
//Definition of the objective function. When testing code remember to change the initial value 
// in the main function.
long double f(std::vector<long double> x)
{
    return exp(x[0]+3*x[1]-0.1) + exp(x[0]-3*x[1] - 0.1) + exp(x[0]-0.1);
    //return pow(x[0],2) + pow(x[1],2) + 3*pow(x[2],2) + pow(x[0],2)*pow(x[1],2);
}


//Computes the numerical gradient using symmetric finite differences.
std::vector<long double> get_gradient(std::vector<long double> x)
{
    //Epsilon is chosen to be small.
    long double epsilon = pow(2,-64);
    std::vector<long double> gradient;
    for(int i = 0; i < x.size(); i++)
    { 
       std::vector<long double> h(x.size(), 0.0);
       //h is scaled with the value of x.
       h[i] = x[i]*sqrt(epsilon);
       //The symmetric finite difference is the limit as h goes to zero of (f(x+h)-f(x-h))/(2h).
       //It is better numerically as it tends to give less error.
       std::vector<long double> x_plus_h = vec_add(x,h);
       std::vector<long double> x_min_h = vec_add(x,scal_mult(-1,h));
       //If the gradient is zero then we will get nan, this makes sure we always return
       // a double.
       if(isnan((f(x_plus_h)-f(x_min_h))/(2*h[i])) == 1)
       {
           gradient.push_back(0.0);
       }
       else
       {
           gradient.push_back((f(x_plus_h)-f(x_min_h))/(2*h[i]));
       }
    }
    return gradient;
}

//Computes the numerical Hessian matrix using symmetric finite differences
std::vector<std::vector<long double>> get_hessian(std::vector<long double> x)
{
    long double epsilon = pow(2,-64);
    //Declaring the matrix as a multidimensional vector.
    std::vector<std::vector<long double>> H;
    //The loop fills the matrix with zeros so we can use element reference.
    for(int i = 0; i < x.size(); i++)
    {
        std::vector<long double> row(x.size(), 0.0);
        H.push_back(row);
    }
    for(int i = 0; i < x.size() ; i++)
    {
        //The Hessian matrix is symmetric, and elements along the diagonal are second 
        //derivatives with respect to one variable. Off diagonal elements are mixed partials.
        std::vector<long double> h(x.size(), 0.0);
        
        //scaling epsilon with x.
        h[i] = x[i]*pow(epsilon,1/4);
        std::vector<long double> x_plus_h = vec_add(x,h);
        std::vector<long double> x_min_h = vec_add(x,scal_mult(-1,h));
        double second_deriv = (f(x_plus_h)-2*f(x) + f(x_min_h))/(pow(h[i],2));
        if(isnan(second_deriv)==0)
        {
           H.at(i).at(i) = second_deriv; 
        }
        //Since the matrix is symmetric, we can calculate the upper triangular portion
        // in order to save a few operations, slightly improving the speed.
        for(int j = i+1; j < x.size(); j++)
        {
            //Calculating the mixed partial with second order finite difference
            // approximation according to the formula in Numerical Reicpes.
            h[j] = x[j]*pow(epsilon,1/4);
            std::vector<long double> x_plus_h = vec_add(x,h);
            std::vector<long double> x_min_h = vec_add(x,scal_mult(-1,h));
            h[j] = -1*h[j];
            std::vector<long double> x_mixed_1 = vec_add(x,h);
            std::vector<long double> x_mixed_2 = vec_add(x,scal_mult(-1,h));
            double mixed_partial = ((f(x_plus_h)-f(x_mixed_1))-(f(x_mixed_2)-f(x_min_h)))/(2*pow(h[i],2) + 2*pow(h[j],2));
            //This line makes sure we always return a double. It will only be nan if the 
            // second order mixed partial is zero, so we can be confident returning zero.
            if(isnan(mixed_partial) == 0)
            {
                H.at(i).at(j) = mixed_partial;
                H.at(j).at(i) = mixed_partial;
            }           
        }
    }
    return H;
}

//Finds the minimum by gradient descent. Gradient descent moves in the direction of 
// steepest decscent by calculating the gradient and moving in the direction of the
// negative gradient. The step size is calculated by backtracking line search.
void gradient_descent(std::vector<long double> x)
{
    std::cout<< "Optimizing via gradient descent..." << std::endl;
    //alpha and beta are constants that determine the exit conditions and the rate at which
    //t decays. It is best to modify them according to the problem.
    double alpha = .1;
    double beta = .7;
    // I define nu as a small number to check that the norm of the gradient is less than.
    //This is to ensure the gradient is sufficiently close to zero.
    long double nu = pow(2,-8);
    std::vector<long double> grad_f{get_gradient(x)};
    //The algorithm moves in the direction of the negative gradient.
    double t = 1;
    //f_tilde and exit are exit conditions provided boyd. 
    double f_tilde = f(vec_add(x,scal_mult(-t,grad_f)));
    double exit = f(x)-alpha*t*pow(lpnorm(grad_f,2),2);
    int count = 0;
    //The loops performs the algorithm until the gradient is sufficiently close to zero.
    //In this case sufficiently close means that the l2 norm of the gradient is less than nu, and that f_tilde > exit, or the fail safe of the algorithm reaching 100 loops, which usually indicates there was some sort of problem.
    while((f_tilde >= exit)||(lpnorm(grad_f,2) > nu ))
    {
        count+=1;
        
        for(int i = 0; i < x.size(); i++)
        {
            if(t*grad_f[i] > 1)
            {
                x[i] = x[i]-1;
            }
            else
            {
            x[i] = x[i]-t*grad_f[i];
            }
        }
        std::vector<long double> new_grad{get_gradient(x)};
        //print_vector(grad_f);
        //print_vector(x);
        for(int i =0; i < new_grad.size(); i++)
        {
            grad_f[i] = new_grad[i];
        }
        exit = f(x)-alpha*t*pow(lpnorm(grad_f,2),2);
        f_tilde = f(vec_add(x,scal_mult(-t,grad_f)));
        //We scale t by beta so the step size decreases.
        t *= beta;
        if(count == 100)
        {
            break;
        }
           
    }
    std::cout << "x*: ";
    print_vector(x);
    std::cout << "f(x*):" << f(x) << std::endl;
    std::cout << "grad_f(x*):";
    print_vector(grad_f);
}

int main()
{
    std::vector<long double> initial_value{4,4};
    gradient_descent(initial_value);
    //Testing the calculation of the Hessian
  
    std::vector<std::vector<long double>> Hessian = get_hessian(initial_value);
    std::cout<< "Hessian Matrix at the initial value:" << std::endl;
    print_matrix(Hessian);
    return 0;
}