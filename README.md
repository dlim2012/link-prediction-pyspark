# link-prediction-pyspark
Here are the instuctions on how to run our code

1. Please navigate to this github link:
https://github.com/sreeja-g/link-prediction-pyspark

2. Download any of the following ipynb/py files and open them in google colab
	* Results_based_on_neighbors.ipynb
	* Results_based_on_path.ipynb
	* based_on_path/shortest_path.py


3. Make sure that you have datasets needed added to the files in 
	* email-Eu-core-temporal-Dept2.txt
	* email-Eu-core-temporal-Dept3.txt

Under the runtime tab, select Run all.

The expiriment results can be found at the bottom of the ipynb files.

And each code cell in the notebook will print out the intermidiate matrix that can be used for testing!

Algorithms Implemented:
1. Common neighbors algorithm: size of the neighbors of x intersected with the neighbors of y
2. Jaccard similarity algorithm: size of the neighbors of x intersected with the neighbors of y divided by the size of the neighbors of x unioned with the neighbors of y
3. Adamic adar algorithm: sum of one divided by the log of the size of neighbors of a node z over the intersection of the neighbors of x and neighbors of y
4. Resource allocation algorithm: sum of one divided by the size of neighbors of a node z over the intersection of the neighbors of x and neighbors of y
5. Preferential attachment algorithm: product of the size of the neighbors of x and the size of the neighbors of y
6. CNGF algorithm: sum of the degree of node z divided by the log of the size of the neighbors of z over the intersection of the neighbors of x and neighbors of y
7. Shortest Path Algorithm: negative one times the length of the shortest path (We used repeated Bellman Ford algorithm for the shortest path)
8. Katz Algorithm: sum from L=1 to infinity of beta to the power of L multiplied by the length of the set of all paths from node x to node y where beta is a value between 0 and 1
9. Hitting Time Algorithm: negative of the expected number of steps from node x to node y using a random walk multiplied by the stationary probability of node x plus the number of steps from node y to node x using a random walk multiplied by the stationary probability of node y
10. Simrank algorithm:max of C multiplied by the transpose of matrix A multiplied by matrix S multiplied by matrix A and I, where
     	* A is the adjacency matrix with the values at (a,b) equal to 1 divided by the in neighbors of node b if there is an edge between nodes a and b and 0 otherwise
     	* C is a constant value between 0 and 1
    	* S is the simrank matrix







