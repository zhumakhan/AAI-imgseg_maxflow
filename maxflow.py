from PIL import Image
from collections import defaultdict as ddict

INF = 1e10
SCALE = 1e5
dirs = ((-1, 0), (0, -1))

def bfs(adj, capacity, s, t, parent):
	parent[:]=-1
	parent[s]=-2;
	queue = []
	queue.append((s, INF))

	while len(queue) > 0:
		u = queue[0][0]
		flow = queue[0][1]

		queue.pop(0)

		for v in adj[u]:
			if parent[v] == -1 and capacity[u][v] > 0:
				parent[v] = u
				new_flow = min(flow, capacity[u][v])
				if v == t:
					return new_flow
				else:
					queue.append((v, new_flow))


	return 0

def maxflow(adj, capacity, s, t, n):
	flow = 0
	parent = [0 for _ in range(n)]

	while True:
		new_flow = bfs(adj, capacity, s, t, parent)
		if new_flow == 0:
			break

		curr = t

		while curr != s:
			prev = parent[curr]
			capacity[prev][curr] -= new_flow
			capacity[curr][prev] += new_flow
			curr = prev


def inference(filepath, lamda, sigma):
	image = Image.open(filepath).convert('L')
	h,w = image.shape

	max_weight = -INF

	adj = [[] for _ in range(h*w)]
	# capacity = [ddict(lamda:0) for _ in range(h*w)]
	capacity = [{} for _ in range(h*w)]

	for i in range(h):
		for j in range(w):
			pi = image[i][j]
			for d in dirs:
				ii = i + d[0]
				jj = j + d[1]
				if ii < 0 or jj < 0:
					continue
				qi = image[ii][jj]
				l2dis = abs(pi-qi)
				n_weight = lamda*exp(-l2dis*l2dis/(2*sigma*sigma))
				n_weight_int = int(n_weight*SCALE)
				adj[i*w+j].append(ii*w+j)
				adj[ii*w+j].append(i*w+j)
				capacity[i*w+j][ii*w+j]=n_weight_int
				capacity[ii*w+j][i*w+j]=n_weight_int
				max_weight = max(max_weight, n_weight_int)

	
	max_weight = SCALE * max_weight
	








