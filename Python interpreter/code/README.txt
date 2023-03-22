Index :
1) Title
2) Configuration required
3) Usage
4) License

Contents : 

1) Title : Python interpreter
    We have implemented basic code in pyhton which serves an interpreter for Python language that does 
divide. We work with tokens, lexical analyzer, and expressions that divide integers from integers. We 
have also added support for dividing float values.

2) Configuration required
    Our code does not make use of any external python libraries.
    Given below is the configuration needed:
        i) .ipynb file:
            To run this file the user needs to open the .ipynb file on a text editor supporting this 
        format. Then the user needs to create input.txt in the same directory which has the input lines
        for our interpreter. Simply clicking on run-all button yeilds the required output.txt result.
        ii) .py file:
            To run this file user must have python3 available. They simply have to create 
        input.txt in the same directory which has the input lines for our interpreter. Simply clicking 
        on run button yeilds the required output.txt result.

3) Usage
    To use this code the user has to enter the lines of code the user wants the interpreter to evaluate.
This code evaluates basic expressions and outputs their result. Our code also is capable of assignment
operations and stores the evaluated value associated with the variable. These values can be called later
in the code by simply writing that variable.
    Input format:  There are two types of input that the code evaluates, which are an evaluation statement and assignment statement.
    In BNF they are written as:
        Input ::= StatementList
        StatementList ::= StatementList Statement | StatementList
        Statement ::= Assignment | Expression
        Assignment ::= Identifier "=" Expresstion
        Expression ::= Expression "/" Expression | "(" Expression ")" | Identifier | Integers

    Integer and Identifier are terminals which can be derived from regular expressions
        Integer = [0-9]+(\.[0-9]+)?
        Identifier = [a-zA-Z_]+[0-9a-zA-Z_]*

4) License
    For evaluation purposes for CSN-352 project-1.
    Made by : Dhruv Shandilya(20114034), Chinayush Waman Wasnik(20114027), Sumit Bera(20114023).