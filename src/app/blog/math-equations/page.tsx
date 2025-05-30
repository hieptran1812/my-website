import BlogReader from "../../components/BlogReader";

export default function MathEquationsPage() {
  const content = `
    <h1>Mathematical Expressions Demo</h1>
    <p>This blog post demonstrates the LaTeX math rendering capabilities in our blog reader.</p>

    <h2>Inline Math Examples</h2>
    <p>Here are some basic inline math expressions: $x = 5$, $y = 2x + 3$, and $\\alpha + \\beta = \\gamma$.</p>
    <p>The quadratic formula is: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$</p>
    <p>Einstein's mass-energy equivalence: $E = mc^2$</p>

    <h2>Display Math Examples</h2>

    <h3>Quadratic Formula</h3>
    <p>The quadratic formula for solving $ax^2 + bx + c = 0$ is:</p>
    $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

    <h3>Integral</h3>
    <p>The fundamental theorem of calculus:</p>
    $$\\int_a^b f'(x) \\, dx = f(b) - f(a)$$

    <h3>Matrix</h3>
    <p>A 2x2 matrix:</p>
    $$A = \\begin{pmatrix}
    a & b \\\\
    c & d
    \\end{pmatrix}$$

    <h3>Complex Expression</h3>
    <p>Euler's identity:</p>
    $$e^{i\\pi} + 1 = 0$$

    <h3>Series</h3>
    <p>The Taylor series of $e^x$:</p>
    $$e^x = \\sum_{n=0}^{\\infty} \\frac{x^n}{n!} = 1 + x + \\frac{x^2}{2!} + \\frac{x^3}{3!} + \\cdots$$

    <h3>Limit</h3>
    <p>Definition of a derivative:</p>
    $$f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$$

    <h3>Greek Letters and Special Symbols</h3>
    <p>Some Greek letters: $\\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\zeta, \\eta, \\theta$</p>
    <p>Set theory symbols: $\\in, \\notin, \\subset, \\subseteq, \\cup, \\cap, \\emptyset$</p>
    <p>Logic symbols: $\\forall, \\exists, \\neg, \\wedge, \\vee, \\implies, \\iff$</p>

    <h3>Number Sets</h3>
    <p>Common number sets: $\\mathbb{N}, \\mathbb{Z}, \\mathbb{Q}, \\mathbb{R}, \\mathbb{C}$</p>
    <p>Which can be written as: $\\NN, \\ZZ, \\QQ, \\RR, \\CC$ using our custom macros.</p>

    <h2>Advanced Examples</h2>

    <h3>Aligned Equations</h3>
    <p>System of linear equations:</p>
    $$\\begin{align}
    2x + 3y &= 7 \\\\
    x - y &= 1
    \\end{align}$$

    <h3>Fractions and Binomial Coefficients</h3>
    <p>Binomial theorem:</p>
    $$(x + y)^n = \\sum_{k=0}^{n} \\binom{n}{k} x^{n-k} y^k$$

    <h3>Calculus</h3>
    <p>Partial derivatives:</p>
    $$\\frac{\\partial^2 f}{\\partial x \\partial y} = \\frac{\\partial}{\\partial x}\\left(\\frac{\\partial f}{\\partial y}\\right)$$

    <h3>Statistics</h3>
    <p>Normal distribution probability density function:</p>
    $$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$$

    <h2>Error Handling</h2>
    <p>This is an intentionally malformed expression to test error handling: $\\invalid{syntax$</p>
    <p>And this one too: $$\\another\\bad\\expression{$$</p>

    <h2>Mixed Content</h2>
    <p>You can mix regular text with inline math like $f(x) = x^2 + 2x + 1$ and then continue with more text.</p>
    <p>The area under the curve $y = x^2$ from $x = 0$ to $x = 1$ is:</p>
    $$\\int_0^1 x^2 \\, dx = \\left[\\frac{x^3}{3}\\right]_0^1 = \\frac{1}{3}$$
    <p>This completes our mathematical expressions demonstration!</p>
  `;

  return (
    <BlogReader
      title="Mathematical Expressions with LaTeX"
      publishDate="2024-12-20"
      readTime="8 min read"
      tags={["mathematics", "latex", "katex", "equations"]}
      category="Tutorial"
      author="Hiep Tran"
      dangerouslySetInnerHTML={{ __html: content }}
    />
  );
}
