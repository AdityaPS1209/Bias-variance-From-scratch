import numpy as np
import matplotlib.pyplot as plt
from manim import *
import math as mt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)  # For reproducibility
X = np.arange(1,25,0.2)  # Random x values between 0 and 10
noise = np.random.normal(0, 0.5, size=X.shape)
Y = 0.06*X**2 + 2*np.sin(X) +noise
degree = 10
l1 = np.arange(0,1,0.003)  # Different regularization strengths
l2=np.arange(1,10,0.5)
l3=np.arange(10,100,2)
l = np.concatenate((l1, l2, l3), axis=0)
# Dictionary to store weights, bias, and lambda values
results = []
X1=X.reshape(-1)
Y1=Y.reshape(-1)
points = list(zip(X1, Y1))
X=X.reshape(-1,1)
# Perform polynomial regression with Ridge (L2 regularization) for each lambda
for alpha in l:
    # Create a pipeline to combine polynomial transformation and Ridge regression
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    model.fit(X, Y)
    
    # Extract model coefficients and intercept
    weights = model.named_steps['ridge'].coef_
    bias = model.named_steps['ridge'].intercept_
    
    # Store the weights, bias, and lambda value in results
    results.append({
        'lambda': alpha,
        'weights': weights,
        'bias': bias
    })





class graph(Scene):

    def construct(self):    
        image = ImageMobject("123.png")
        text = Text("Data Dissection")
        text.scale(0.3)
        image.to_edge(UP)
        text.next_to(image, DOWN)
        text.shift(UP)
        image.scale(0.3)
        image.shift(RIGHT*2)
        text.shift(RIGHT*2)
        self.add(image,text)
        
        # Adding Lagrange equation
        lambda_val = ValueTracker(0)
        lagrange_text = MathTex(r"\mathcal{L}(x, y, \lambda) = f(x, y) + ")
        lambda_term = MathTex(r"\lambda").set_color(YELLOW)  # Highlight lambda for visibility
        constraint_term = MathTex(r"\cdot g(x, y)")

        # Dynamic lambda value that updates within the equation
        lambda_value = DecimalNumber(lambda_val.get_value()).set_color(YELLOW)
        # Set the lambda value to update as per lambda_val tracker
        lambda_value.add_updater(lambda m: m.set_value(lambda_val.get_value()))
        self.add(lagrange_text, lambda_value, constraint_term)
        # defines the axes 
        axes = Axes(x_range=[1, 25,1], y_range=[1, 40,6], x_length=7, y_length=7)
        func = axes.plot(lambda x: x, color=BLUE)
        dot = Dot(radius=0.3)
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=RED) for x, y in points
        ])
        # Add all parts to the scene
        grph=Group(axes, func,dots)
        grph.scale(0.7)
        grph.shift(LEFT*3)
        lagrange_text.next_to(grph, RIGHT)
        lambda_term.next_to(lagrange_text, RIGHT)
        lambda_value.scale(1).next_to(lagrange_text, RIGHT)
        constraint_term.next_to(lambda_value, RIGHT)
        self.add(grph)  
        k=0
        for r in results:
            k+=1
            if(k<4):
                w = r["weights"].tolist()
                lambda_val.set_value(r["lambda"])
                func2 = axes.plot(lambda x: w[1]*x + w[2]*x**2 +
                            w[3]*x**3 + w[4]*x**4 + w[5]*x**5 +
                            w[6]*x**6 + w[7]*x**7 + w[8]*x**8 +
                            w[9]*x**9 + w[10]*x**10 + r["bias"], color=BLUE)
                self.play(Transform(func,func2))
            if(k>334 and k<339):
                #self.wait(2)
                w = r["weights"].tolist()
                lambda_val.set_value(r["lambda"])
                func2 = axes.plot(lambda x: w[1]*x + w[2]*x**2 +
                            w[3]*x**3 + w[4]*x**4 + w[5]*x**5 +
                            w[6]*x**6 + w[7]*x**7 + w[8]*x**8 +
                            w[9]*x**9 + w[10]*x**10 + r["bias"], color=BLUE)
                self.play(Transform(func,func2))
            if(k>392):
                #self.wait(2)
                w = r["weights"].tolist()
                lambda_val.set_value(r["lambda"])
                func2 = axes.plot(lambda x: w[1]*x + w[2]*x**2 +
                            w[3]*x**3 + w[4]*x**4 + w[5]*x**5 +
                            w[6]*x**6 + w[7]*x**7 + w[8]*x**8 +
                            w[9]*x**9 + w[10]*x**10 + r["bias"], color=BLUE)
                self.play(Transform(func,func2))
        self.wait(1)
        

    