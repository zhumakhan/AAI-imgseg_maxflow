from PIL import Image

INF = 1e10

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


def im2graph(filepath):
	image = Image.open(filepath).convert('L')
	h,w = image.shape
	






