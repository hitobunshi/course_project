import func
from gradient_descent import GradientDescent
from lr_scheduler import ConstLrScheduler


if __name__ == "__main__":
    lr_scheduler = ConstLrScheduler(1e-4)
    gradient_descent = GradientDescent(lr_scheduler, func.sin_cos, func.sin_cos_grad, num_args=2, tol=1e-8)
    min_point = gradient_descent.descent()
    print(min_point, func.mean_quad(min_point))
