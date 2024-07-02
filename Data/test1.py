from manim import *

class SolveQuadraticEquation(Scene):
    def construct(self):
        # Step 1: Display the quadratic equation
        equation = MathTex("ax^2 + bx + c = 0")
        equation.to_edge(UP)
        self.play(Write(equation))
        self.wait(1)
        
        # Step 2: Display the quadratic formula
        formula = MathTex("x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{2a}")
        formula.next_to(equation, DOWN, buff=1)
        self.play(Write(formula))
        self.wait(2)
        
        # Step 3: Highlight the discriminant
        discriminant = MathTex("b^2 - 4ac")
        discriminant.next_to(formula, DOWN, buff=1)
        self.play(Write(discriminant))
        self.wait(1)
        
        # Step 4: Explain each part of the formula
        b2_part = MathTex("b^2").set_color(YELLOW).move_to(discriminant.get_left() + 0.5 * RIGHT)
        minus4ac_part = MathTex("- 4ac").set_color(RED).move_to(discriminant.get_right() - 0.5 * RIGHT)
        
        self.play(ReplacementTransform(discriminant[0:2].copy(), b2_part))
        self.wait(1)
        self.play(ReplacementTransform(discriminant[2:].copy(), minus4ac_part))
        self.wait(1)
        
        # Step 5: Solve the discriminant
        self.play(FadeOut(b2_part), FadeOut(minus4ac_part))
        
        discriminant_solution = MathTex("D = b^2 - 4ac")
        discriminant_solution.next_to(formula, DOWN, buff=1)
        self.play(Transform(discriminant, discriminant_solution))
        self.wait(2)
        
        # Step 6: Show the final solution
        solution = MathTex(
            "x = \\frac{{-b + \\sqrt{D}}}{2a}", "\\text{ or }", "x = \\frac{{-b - \\sqrt{D}}}{2a}"
        )
        solution.next_to(discriminant_solution, DOWN, buff=1)
        self.play(Write(solution))
        self.wait(2)

        # Ending
        final_text = Text("Solution to the Quadratic Equation").next_to(solution, DOWN, buff=1)
        self.play(Write(final_text))
        self.wait(2)
