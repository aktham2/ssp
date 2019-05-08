import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 
'''
using a topological sorting algorithm. complexity is O(N+E)
MDP with optimality criterion of minimal cost from source node to destination node
different MD control policies are explored. 
Markovian because all past information is incorporated into current state as the current node and current cost
		Decision rules: Optimal expectation, variance, worst case, ...
Probability distribution on the cost of each edge.
introduce action set, with a prob dist on the edge taken weighted more toward s_i for a_i
'''
class Graph: 
	def __init__(self,nodes): 
		self.N = nodes 
		self.graph = defaultdict(list)
		# dict for node -> (destination,weight)
	def addEdge(self,u,v,w,p):
		# edge u->v, weight w = [w_1,w_2,...] p=[p_1,p_2,...] where sum(p_i)=1 and p_i>0
		# later can make v=[] as well
		self.graph[u].append((v,w,p))
	def traverse(self,S):
		# traverse the DAG with sequence S, if possible, output cost
		# print('\n\n Test: S =',S)
		cost = []
		for i in range(len(S)-1):
			for node,w_s,p_s in self.graph[S[i]]:
				if S[i+1] == node:
					cost.append( np.random.choice(w_s,p=p_s))
					break
			if len(cost) < i+1:
				print('Path not possible')
				return
		# print('\tcost =', cost)
		# print('\ttotal = ', sum(cost))
		return sum(cost)
	
	def mcSim(self,S):
		x = []
		for i in range(10000):
			x.append(self.traverse(S))
		return x
		
	def topologicalSortUtil(self,v,visited,stack):
		# sorts nodes from destination -> source (DAGs only)
		visited[v] = True
		if v in self.graph.keys():
			for node,w_s,p_s in self.graph[v]:
				if visited[node] == False:
					self.topologicalSortUtil(node,visited,stack)
		stack.append(v) 

	def shortestPath_E(self, s):
		# OPTIMAL EXPECTATION (min expected value)
		# to record the state sequence
		predecessor = [0]*self.N
		visited = [False]*self.N
		stack =[]
		# Sort starting from source
		for i in range(self.N): 
			if visited[i] == False: 
				self.topologicalSortUtil(s,visited,stack)
		# destination is the top of the sorted list
		d = stack[0]
		# Initialize distances to all nodes as infinite and 
		# distance to source as 0 
		cost = [float("Inf")]*self.N
		cost[s] = 0
		worst = [float("Inf")]*self.N
		worst[s] = 0
		p_worst = 1
		best = [float("Inf")]*self.N
		best[s] = 0
		var = [float("Inf")]*self.N
		var[s] = 0
		while stack: 
			# Furthest state is the source node
			i = stack.pop() 
			# Update distances of all adjacent
			for node,w_s,p_s in self.graph[i]:
				# print(w_s,p_s)
				if cost[node] > cost[i] + np.average(w_s,weights=p_s): 
					cost[node] = cost[i] + np.average(w_s,weights=p_s)
					worst[node] = worst[i] + max(w_s)
					best[node] = best[i] + min(w_s)
					var[node] = var[i] + (sum([x**2*p_s[i] for i,x in enumerate(w_s)])-np.average(w_s,weights=p_s)**2)
					# print(i,node)
					predecessor[node] = i
		node = predecessor[-1]
		S = [d,node]
		while node > 0:
			node = predecessor[node]
			S.append(node)
		S.reverse()
		print('\n Min E[cost]:')
		# print ('The optimal path from ',s, ' to ', d, 'is: ') 
		print('\tS* = ',S)
		print('\tE[cost] = %.2f'%cost[-1])
		print('\tVariance = %.2f'% var[-1])
		print('\tsd = %.2f'% var[-1]**0.5)
		print('\tworst case = %.2f'% worst[-1])
		print('\tbest case = %.2f'%best[-1])
		# print(cost)
		# print(var)
		return S,cost[-1],var[-1],worst[-1],best[-1]
		
	def shortestPath_var(self,s):
		K = 0.5
		# OPTIMAL RISK (min E[cost]+(var(cost))
		#####################
		## CONSIDER WEIGHTS #
		#####################
		# to record the state sequence
		predecessor = [0]*self.N
		visited = [False]*self.N
		stack =[]
		for i in range(self.N): 
			if visited[i] == False: 
				self.topologicalSortUtil(s,visited,stack)
		d = stack[0]
		# init cost and var lists
		cost = [float("Inf")]*self.N
		cost[s] = 0
		worst = [float("Inf")]*self.N
		worst[s] = 0
		p_worst = 1
		best = [float("Inf")]*self.N
		best[s] = 0
		var = [float("Inf")]*self.N
		var[s] = 0
		while stack: 
			# Furthest state is the source node
			i = stack.pop() 
			# Update distances of all adjacent
			for node,w_s,p_s in self.graph[i]:
				var_t = (sum([x**2*p_s[i] for i,x in enumerate(w_s)])-np.average(w_s,weights=p_s)**2)
				# if cost[node]+bias > cost[i] + np.average(w_s,weights=p_s): 
					# if var[node] > var[i] + var_t: 
					# *** subbed this for below statement ***
				# print(node , var[node]**0.5)
				# print(cost[node]+(var[node])*K)
				if cost[node]+(var[node])*K > cost[i] + np.average(w_s,weights=p_s) + var[i]*K + var_t*K: 
					worst[node] = worst[i] + max(w_s)
					best[node] = best[i] + min(w_s)
					cost[node] = cost[i] + np.average(w_s,weights=p_s)
					var[node] = var[i] + var_t
					# print(i,node)
					predecessor[node] = i
				# print(node , var[node]**0.5)
		node = predecessor[-1]
		S = [d,node]
		while node > 0:
			node = predecessor[node]
			S.append(node)
		S.reverse()
		# print('\n Min E[cost] within ',bias,'and min var(cost):')
		print('\n Min E[cost]+var(cost):')
		# print ('The optimal path from ',s, ' to ', d, 'is: ') 
		print('\tS* = ',S)
		print('\tE[cost] = %.2f'%cost[-1])
		print('\tVariance = %.2f'% var[-1])
		print('\tsd = %.2f'% var[-1]**0.5)
		print('\tworst case = %.2f'%worst[-1])
		print('\tbest case = %.2f'%best[-1])
		# print(cost)
		# print(var)
		return S,cost[-1],var[-1],worst[-1],best[-1]
	def shortestPath_worstCase(self,s):
		# OPTIMAL RISK (min worst case)
		# to record the state sequence
		predecessor = [0]*self.N
		visited = [False]*self.N
		stack =[]
		for i in range(self.N): 
			if visited[i] == False: 
				self.topologicalSortUtil(s,visited,stack)
		d = stack[0]
		# init cost and var lists
		cost = [float("Inf")]*self.N
		cost[s] = 0
		worst = [float("Inf")]*self.N
		worst[s] = 0
		p_worst = 1
		best = [float("Inf")]*self.N
		best[s] = 0
		# p_worst = 1
		var = [float("Inf")]*self.N
		var[s] = 0
		while stack: 
			# Furthest state is the source node
			i = stack.pop() 
			# Update distances of all adjacent
			for node,w_s,p_s in self.graph[i]:
				# print(cost)
				if worst[node] > worst[i] + max(w_s): 
					worst[node] = worst[i] + max(w_s)
					best[node] = best[i] + min(w_s)
					p_worst *= p_s[np.argmax(w_s)]
					cost[node] = cost[i] + np.average(w_s,weights=p_s)
					var[node] = var[i] + (sum([x**2*p_s[i] for i,x in enumerate(w_s)])-np.average(w_s,weights=p_s)**2)
					# print(i,node)
					predecessor[node] = i
		node = predecessor[-1]
		S = [d,node]
		while node > 0:
			node = predecessor[node]
			S.append(node)
		S.reverse()
		print ('\n Min max(cost)') 
		print('\tS* = ',S)
		# print('\tMax cost = %.2f wp %.4f'% (worst[-1],p_worst))
		print('\tE[cost] = %.2f'%cost[-1])
		print('\tVariance = %.2f'% var[-1])
		print('\tsd = %.2f'% var[-1]**0.5)
		print('\tworst case = %.2f'%worst[-1])
		print('\tbest case = %.2f'%best[-1])
		# print(cost)
		# print(var)
		return S,cost[-1],var[-1],worst[-1],best[-1]
		
''' Random example that works to show difference in policies '''
# g = Graph(6)
# g.addEdge(0, 1, [2,3,1],[0.35,0.55,0.1])
# g.addEdge(0, 2, [3,4,7],[0.15,0.75,0.1])
# g.addEdge(1, 3, [0,2,4],[0.25,0.55,0.2])
# g.addEdge(1, 2, [-2,2,3],[0.1,0.5,0.4])
# g.addEdge(2, 4, [1,5,8],[0.4,0.1,0.5])
# g.addEdge(2, 5, [-20,5,50],[0.25,0.65,0.1])
# g.addEdge(2, 3, [-1,1,5],[0.4,0.5,0.1])
# g.addEdge(3, 4, [-7,2,15],[0.4,0.1,0.5])
# g.addEdge(3, 5, [3,5,20],[0.4,0.55,0.05])
# g.addEdge(4, 5, [-3,2,20],[0.05,0.1,0.85])
''' -------------------------------------------------------- '''

''' 				Daily commute example 				     '''
g = Graph(8)
# home -> train station, wait time for train arrival
g.addEdge(0, 1, [3,5,8],[0.5,0.4,0.1])
# train ride -> work, small probability of delay
g.addEdge(1, 7, [35,65],[0.98,0.02])
# home -> bike -> halfway to work. small risk of injury
g.addEdge(0, 2, [20,25],[0.99,0.01])
# halfway to work -> bike -> work, small risk of injury
g.addEdge(2, 7, [20,28],[0.99,0.01])
# home -> car, to decision of backroad/highway
g.addEdge(0, 4, [6,8,10],[0.2,0.4,0.4])
# decision backroad -> halfway to work. average longer, good worst case, less impact of traffic
g.addEdge(4, 5, [20,25,30],[0.2,0.6,0.2])
# decision highway -> halfway to work, average shorter, good best case, more impact of traffic
g.addEdge(4, 6, [10,15,40],[0.05,0.85,0.1])
# halfway on highway -> work
g.addEdge(6, 7, [5,10,20], [0.05,0.8,0.15])
# change from highway to backroad (near an exit)
g.addEdge(6, 5, [2,3],[0.5,0.5])
# halfway on backroad -> work
g.addEdge(5, 7, [10,12,15],[0.2,0.6,0.2])
# halfway to work -> bike -> TS2
g.addEdge(2, 3, [2,3],[0.25,0.75])
# halfway on BR -> TS2
g.addEdge(5, 3, [2,3],[0.5,0.5])
# halfway on HW -> TS2
g.addEdge(6, 3, [5,6],[0.5,0.5])
# TS2 -> work. all other paths can take this one after halfway
g.addEdge(3, 7, [15,27],[0.98,0.02])
# change mind from the train station to drive
g.addEdge(1, 4, [5,6],[0.5,0.5])

s = 0
print()
S1,mean1,var1,worst1,best1 = g.shortestPath_E(s) 
S2,mean2,var2,worst2,best2 = g.shortestPath_var(s)
S3,mean3,var3,worst3,best3 = g.shortestPath_worstCase(s)

g.traverse(S1)
g.traverse(S2)
g.traverse(S3)

bins = list(range(int(min([best1,best2,best3])),int(max([worst1,worst2,worst3])),1))
plt.hist(g.mcSim(S1),color=(0, 0.1, 0.6, 0.5), bins=bins)
plt.hist(g.mcSim(S2),color=(0, 0.4, 0, 0.7),bins=bins)
plt.hist(g.mcSim(S3),color=(0.8, 0.5, 0, 0.5),bins=bins)
plt.legend(['min( E[cost] )','min( E[cost]+var(cost) )', 'min( max(cost) )'])
plt.title('MC Simulation, 10,000 trials')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.show()
