import func
from gradient_descent import GradientDescent
from lr_scheduler import ConstLrScheduler


if __name__ == "__main__":
    lr_scheduler = ConstLrScheduler(1e-10)
    gradient_descent = GradientDescent(lr_scheduler, func.mean_quad, func.mean_quad_grad, num_args=2)
    print(gradient_descent.descent())
