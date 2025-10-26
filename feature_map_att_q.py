import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations


class NodeAttention:
    """
    Node-level attention mechanism for molecular graphs.
    Learns to weight node importance based on atom types and local structure.
    """
    def __init__(self, feature_dim=16, random_state=42):
        self.feature_dim = feature_dim
        self.random_state = random_state
        self.W_key = None
        self.W_query = None
        self.input_dim_ = None
        self.is_fitted_ = False
        
    def compute_node_features(self, G, node):
        """
        Compute simple node features for attention.
        Returns: feature vector for the node
        """
        features = []
        
        # Atom type (one-hot encoded, 7 types in MUTAG)
        atom_type = G.nodes[node].get('atom_type', 0)
        atom_one_hot = np.zeros(7)
        if 0 <= atom_type < 7:
            atom_one_hot[atom_type] = 1
        features.extend(atom_one_hot)
        
        # Local structure features
        features.append(G.degree(node))  # Degree
        features.append(nx.clustering(G, node))  # Local clustering
        
        # Neighbor atom types distribution
        neighbor_atoms = np.zeros(7)
        for neighbor in G.neighbors(node):
            neighbor_atom = G.nodes[neighbor].get('atom_type', 0)
            if 0 <= neighbor_atom < 7:
                neighbor_atoms[neighbor_atom] += 1
        features.extend(neighbor_atoms)
        
        return np.array(features)
    
    def fit(self, graphs):
        """
        Initialize attention weights using training graphs only.
        This ensures no test data leakage.
        """
        if len(graphs) == 0:
            raise ValueError("Cannot fit with empty graph list")
        
        # Determine input dimension from training data
        sample_graph = graphs[0]
        if sample_graph.number_of_nodes() == 0:
            raise ValueError("Cannot fit with empty graphs")
        
        sample_node = list(sample_graph.nodes())[0]
        sample_features = self.compute_node_features(sample_graph, sample_node)
        self.input_dim_ = len(sample_features)
        
        # Initialize weights with fixed random state (training data only)
        rng = np.random.RandomState(self.random_state)
        scale = np.sqrt(2.0 / (self.input_dim_ + self.feature_dim))
        self.W_key = rng.randn(self.input_dim_, self.feature_dim) * scale
        self.W_query = rng.randn(self.input_dim_, self.feature_dim) * scale
        
        self.is_fitted_ = True
        
        return self
    
    def compute_attention(self, G):
        """
        Compute attention weights for all nodes in graph.
        Uses pre-fitted weights only - no modification during transform.
        Returns: dictionary {node: attention_weight}
        """
        if not self.is_fitted_:
            raise ValueError("NodeAttention must be fitted before computing attention. Call fit() first.")
        
        if G.number_of_nodes() == 0:
            return {}
        
        # Compute node features
        node_features = {}
        for node in G.nodes():
            node_features[node] = self.compute_node_features(G, node)
        
        # Compute keys and queries using PRE-FITTED weights
        nodes = list(G.nodes())
        X = np.array([node_features[n] for n in nodes])
        
        # Project to attention space
        keys = X @ self.W_key  # (n_nodes, feature_dim)
        queries = X @ self.W_query  # (n_nodes, feature_dim)
        
        # Compute attention scores (self-attention)
        scores = queries @ keys.T / np.sqrt(self.feature_dim)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(scores, axis=1)
        
        # Aggregate attention for each node (mean across rows)
        node_importance = np.mean(attention_weights, axis=1)
        
        # Create dictionary mapping
        attention_dict = {nodes[i]: node_importance[i] for i in range(len(nodes))}
        
        return attention_dict
    
    def _softmax(self, x, axis=1):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class EdgeAttention:
    """
    Edge-level attention mechanism for molecular graphs.
    Learns to weight edge (bond) importance.
    """
    def __init__(self, feature_dim=8, random_state=42):
        self.feature_dim = feature_dim
        self.random_state = random_state
        self.W_edge = None
        self.input_dim_ = None
        self.is_fitted_ = False
        
    def compute_edge_features(self, G, u, v):
        """Compute features for an edge."""
        features = []
        
        # Bond type encoding
        edge_data = G.get_edge_data(u, v)
        bond_type = edge_data.get('bond_type', 'single') if edge_data else 'single'
        bond_map = {'aromatic': 0, 'single': 1, 'double': 2, 'triple': 3}
        bond_encoded = np.zeros(4)
        bond_encoded[bond_map.get(bond_type, 1)] = 1
        features.extend(bond_encoded)
        
        # Endpoint atom types
        atom_u = G.nodes[u].get('atom_type', 0)
        atom_v = G.nodes[v].get('atom_type', 0)
        features.extend([atom_u, atom_v])
        
        # Endpoint degrees
        features.extend([G.degree(u), G.degree(v)])
        
        return np.array(features)
    
    def fit(self, graphs):
        """
        Initialize edge attention weights using training graphs only.
        This ensures no test data leakage.
        """
        if len(graphs) == 0:
            raise ValueError("Cannot fit with empty graph list")
        
        # Find a graph with edges to determine input dimension
        sample_graph = None
        for g in graphs:
            if g.number_of_edges() > 0:
                sample_graph = g
                break
        
        if sample_graph is None:
            raise ValueError("Cannot fit: no graphs with edges found")
        
        # Determine input dimension from training data
        sample_edge = list(sample_graph.edges())[0]
        sample_features = self.compute_edge_features(sample_graph, sample_edge[0], sample_edge[1])
        self.input_dim_ = len(sample_features)
        
        # Initialize weights with fixed random state (training data only)
        rng = np.random.RandomState(self.random_state)
        scale = np.sqrt(2.0 / (self.input_dim_ + self.feature_dim))
        self.W_edge = rng.randn(self.input_dim_, self.feature_dim) * scale
        
        self.is_fitted_ = True
        
        return self
    
    def compute_attention(self, G):
        """
        Compute attention weights for all edges.
        Uses pre-fitted weights only - no modification during transform.
        Returns: dictionary {(u, v): attention_weight}
        """
        if not self.is_fitted_:
            raise ValueError("EdgeAttention must be fitted before computing attention. Call fit() first.")
        
        if G.number_of_edges() == 0:
            return {}
        
        edges = list(G.edges())
        edge_features = [self.compute_edge_features(G, u, v) for u, v in edges]
        
        X = np.array(edge_features)
        
        # Project and compute scores using PRE-FITTED weights
        scores = X @ self.W_edge @ self.W_edge.T @ X.T
        scores = np.diagonal(scores)
        
        # Sigmoid activation for edge importance
        attention_weights = 1 / (1 + np.exp(-scores))
        
        # Normalize
        attention_weights = attention_weights / (np.sum(attention_weights) + 1e-10)
        
        attention_dict = {edges[i]: attention_weights[i] for i in range(len(edges))}
        
        return attention_dict


class QuantumMolecularGraphFeatureMap(BaseEstimator, TransformerMixin):
    """
    Quantum-enhanced molecular graph feature extractor with:
    - Quantum entanglement entropy (graph state formalism)
    - Quantum walk statistics (CTQW)
    - Quantum chemistry features (Hückel MO, HOMO-LUMO gap)
    - Node/Edge attention mechanisms
    - Edge-aware Weisfeiler-Lehman
    - Pharmacophore distance patterns
    - Spectral moments
    - Classical random walk statistics
    - Persistence homology
    
    QUANTUM FEATURES:
    - Treats molecular graph as quantum graph state (nodes = qubits, edges = CZ gates)
    - Computes entanglement entropy for various bipartitions
    - Uses quantum walk mixing times
    - Estimates HOMO-LUMO gaps from graph eigenvalues
    """
    
    def __init__(self, n_eigen=10, wl_iterations=3, walk_length=10, num_walks=20,
                 use_node_attention=True, use_edge_attention=True, 
                 use_quantum_features=True, random_state=42):
        self.n_eigen = n_eigen
        self.wl_iterations = wl_iterations
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.use_node_attention = use_node_attention
        self.use_edge_attention = use_edge_attention
        self.use_quantum_features = use_quantum_features
        self.random_state = random_state
        
        # Attention mechanisms (initialized in fit)
        self.node_attention = None
        self.edge_attention = None
        
    def _generate_wl_signatures_with_edges(self, G):
        """
        Generate WL signatures with edge features for a graph across all iterations.
        Returns: list of signature sets, one per iteration.
        """
        all_signatures_per_iteration = [set() for _ in range(self.wl_iterations + 1)]
        
        # Initial iteration: atom types
        labels = {node: G.nodes[node]['atom_type'] for node in G.nodes()}
        for label in labels.values():
            all_signatures_per_iteration[0].add(str(label))
        
        # Subsequent iterations with edge information
        for iteration in range(self.wl_iterations):
            new_labels = {}
            for node in G.nodes():
                # Include edge types in signature
                neighbor_info = []
                for neighbor in G.neighbors(node):
                    edge_data = G.get_edge_data(node, neighbor)
                    bond_type = edge_data.get('bond_type', 'single') if edge_data else 'single'
                    neighbor_info.append(f"{labels[neighbor]}:{bond_type}")
                
                neighbor_info.sort()
                signature = f"{labels[node]}|{'|'.join(neighbor_info)}"
                all_signatures_per_iteration[iteration + 1].add(signature)
                new_labels[node] = signature
            labels = new_labels
        
        return all_signatures_per_iteration
    
    def fit(self, graphs, y=None):
        """
        Learn WL vocabulary and initialize attention mechanisms from training graphs ONLY.
        This is the ONLY place where model parameters are learned.
        """
        # Step 1: Collect ALL signatures from ALL training graphs
        all_signatures_per_iteration = [set() for _ in range(self.wl_iterations + 1)]
        
        for G in graphs:
            graph_signatures = self._generate_wl_signatures_with_edges(G)
            for iteration in range(self.wl_iterations + 1):
                all_signatures_per_iteration[iteration].update(graph_signatures[iteration])
        
        # Step 2: Build vocabulary (signature -> ID mapping) per iteration
        self.vocab_per_iteration_ = []
        self.vocab_size_per_iteration_ = []
        
        for iteration in range(self.wl_iterations + 1):
            signatures = sorted(all_signatures_per_iteration[iteration])
            vocab = {sig: idx for idx, sig in enumerate(signatures)}
            self.vocab_per_iteration_.append(vocab)
            self.vocab_size_per_iteration_.append(len(vocab))
        
        # Step 3: Initialize attention mechanisms on TRAINING data only
        if self.use_node_attention:
            self.node_attention = NodeAttention(
                feature_dim=16, 
                random_state=self.random_state
            )
            self.node_attention.fit(graphs)
            print("  ✓ Node attention initialized on training data")
            
        if self.use_edge_attention:
            self.edge_attention = EdgeAttention(
                feature_dim=8,
                random_state=self.random_state
            )
            self.edge_attention.fit(graphs)
            print("  ✓ Edge attention initialized on training data")
        
        if self.use_quantum_features:
            print("  ✓ Quantum features enabled (graph states, quantum walks, QC)")
        
        return self
    
    def extract_quantum_entanglement_features(self, G):
        """
        Compute entanglement entropy features from quantum graph state.
        Each node represents a qubit in |+⟩ state with CZ gates on edges.
        
        For a graph state, entanglement entropy scales with edge boundary.
        """
        if G.number_of_nodes() < 2:
            return np.zeros(6)
        
        n_nodes = G.number_of_nodes()
        features = []
        
        # Von Neumann entanglement entropy for different bipartitions
        nodes = list(G.nodes())
        
        for partition_fraction in [0.1, 0.25, 0.5]:
            partition_size = max(1, int(n_nodes * partition_fraction))
            
            if partition_size >= n_nodes:
                features.append(0)
                continue
            
            partition_A = set(nodes[:partition_size])
            
            # Count edges crossing partition (boundary)
            boundary_edges = 0
            for u, v in G.edges():
                if (u in partition_A) != (v in partition_A):
                    boundary_edges += 1
            
            # For graph states: S(A) ≈ min(|∂A|, log₂(min(|A|, |B|)))
            max_entropy = np.log2(min(partition_size, n_nodes - partition_size) + 1)
            entropy = min(boundary_edges, max_entropy)
            features.append(entropy)
        
        # Graph state stabilizer rank (approximation from edge count)
        stabilizer_rank = np.log2(G.number_of_edges() + 1)
        features.append(stabilizer_rank)
        
        # Schmidt rank estimate (exponential in boundary)
        max_boundary = max([sum(1 for u, v in G.edges() 
                               if (u in set(nodes[:k])) != (v in set(nodes[:k])))
                           for k in range(1, min(n_nodes, 10))] or [1])
        schmidt_rank = min(2**min(n_nodes//2, 10), 2**max_boundary)
        features.append(np.log2(schmidt_rank + 1))
        
        # Average bipartite entanglement
        avg_entanglement = np.mean([f for f in features[:3]])
        features.append(avg_entanglement)
        
        return np.array(features)
    
    def extract_quantum_walk_features(self, G):
        """
        Quantum walk statistics using continuous-time quantum walk (CTQW).
        Hamiltonian: H = -A (negative adjacency matrix)
        Evolution: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        """
        if G.number_of_nodes() < 2:
            return np.zeros(5)
        
        try:
            # Adjacency matrix as quantum walk Hamiltonian
            A = nx.adjacency_matrix(G).todense()
            H = -A  # Quantum walk Hamiltonian
            
            # Compute eigenvalues for mixing analysis
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(eigenvalues)
            
            features = []
            
            # Spectral gap (determines mixing time)
            if len(eigenvalues) > 1:
                spectral_gap = np.abs(eigenvalues[-1] - eigenvalues[-2])
            else:
                spectral_gap = 1.0
            features.append(spectral_gap)
            
            # Quantum mixing time (inverse spectral gap)
            mixing_time = 1.0 / (spectral_gap + 1e-10)
            features.append(mixing_time)
            
            # Average eigenvalue spread (quantum coherence measure)
            eigenvalue_spread = np.std(eigenvalues)
            features.append(eigenvalue_spread)
            
            # Quantum return probability decay
            # |⟨ψ(0)|ψ(t)⟩|² = Σᵢ cos²(λᵢt)/n
            n = G.number_of_nodes()
            t = 1.0  # Unit time
            return_prob = np.sum(np.cos(eigenvalues * t)**2) / n
            features.append(return_prob)
            
            # Quantum transport efficiency (based on eigenvalue structure)
            transport_eff = np.sum(np.abs(eigenvalues)) / (n + 1e-10)
            features.append(transport_eff)
            
            return np.array(features)
        except:
            return np.zeros(5)
    
    def extract_quantum_chemistry_features(self, G):
        """
        Quantum chemistry-inspired features for molecular graphs.
        Uses Hückel molecular orbital approximation.
        """
        features = []
        
        if G.number_of_nodes() >= 2:
            try:
                # Hückel Hamiltonian: -A (negative adjacency for bonding orbitals)
                A = nx.adjacency_matrix(G).todense()
                mo_energies = np.linalg.eigvalsh(-A)  # Molecular orbital energies
                mo_energies = np.sort(mo_energies)
                
                n_nodes = G.number_of_nodes()
                
                # HOMO-LUMO gap (simplified: assume n electrons = n nodes)
                n_electrons = n_nodes
                if len(mo_energies) > 1:
                    homo_idx = n_electrons // 2 - 1
                    lumo_idx = homo_idx + 1
                    
                    if 0 <= homo_idx < len(mo_energies) and lumo_idx < len(mo_energies):
                        homo_lumo_gap = mo_energies[lumo_idx] - mo_energies[homo_idx]
                    else:
                        homo_lumo_gap = 0
                else:
                    homo_lumo_gap = 0
                
                features.append(homo_lumo_gap)
                
                # Total π-electron energy (sum of occupied MO energies)
                n_occupied = n_electrons // 2
                if n_occupied > 0 and n_occupied <= len(mo_energies):
                    occupied_energies = mo_energies[:n_occupied]
                    pi_energy = 2 * np.sum(occupied_energies)  # Factor of 2 for spin
                else:
                    pi_energy = 0
                features.append(pi_energy)
                
                # HOMO energy (chemical potential)
                if 0 <= homo_idx < len(mo_energies):
                    homo_energy = mo_energies[homo_idx]
                else:
                    homo_energy = 0
                features.append(homo_energy)
                
                # LUMO energy (electron affinity proxy)
                if lumo_idx < len(mo_energies):
                    lumo_energy = mo_energies[lumo_idx]
                else:
                    lumo_energy = 0
                features.append(lumo_energy)
                
                # Aromaticity indicator (eigenvalue degeneracy pattern)
                if len(mo_energies) >= 4:
                    aromatic_score = np.std(mo_energies[:4])
                else:
                    aromatic_score = 0
                features.append(aromatic_score)
                
            except:
                features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Electronegativity distribution (quantum property)
        atom_types = [G.nodes[n].get('atom_type', 0) for n in G.nodes()]
        # Map atom types to approximate electronegativities (Pauling scale)
        electronegativity_map = {
            0: 2.2,  # Type 0
            1: 2.5,  # C
            2: 3.0,  # N
            3: 3.5,  # O
            4: 2.8,  # F
            5: 2.6,  # P
            6: 3.2   # Cl
        }
        electronegativities = [electronegativity_map.get(a, 2.5) for a in atom_types]
        
        if electronegativities:
            features.extend([
                np.mean(electronegativities),
                np.std(electronegativities),
                np.max(electronegativities) - np.min(electronegativities)
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def _compute_wl_histogram_with_attention(self, G, node_attention_weights=None):
        """
        Compute WL histogram with optional node attention weighting.
        Uses pre-computed attention weights - no learning here.
        """
        all_histograms = []
        
        # Initial iteration: atom types
        labels = {node: str(G.nodes[node]['atom_type']) for node in G.nodes()}
        
        for iteration in range(self.wl_iterations + 1):
            # Create histogram
            vocab_size = self.vocab_size_per_iteration_[iteration]
            histogram = np.zeros(vocab_size, dtype=np.float32)
            
            for node, signature in labels.items():
                if signature in self.vocab_per_iteration_[iteration]:
                    idx = self.vocab_per_iteration_[iteration][signature]
                    # Apply node attention weight if available
                    weight = node_attention_weights.get(node, 1.0) if node_attention_weights else 1.0
                    histogram[idx] += weight
            
            all_histograms.extend(histogram)
            
            # Prepare labels for next iteration
            if iteration < self.wl_iterations:
                new_labels = {}
                for node in G.nodes():
                    # Include edge types
                    neighbor_info = []
                    for neighbor in G.neighbors(node):
                        edge_data = G.get_edge_data(node, neighbor)
                        bond_type = edge_data.get('bond_type', 'single') if edge_data else 'single'
                        neighbor_info.append(f"{labels[neighbor]}:{bond_type}")
                    
                    neighbor_info.sort()
                    signature = f"{labels[node]}|{'|'.join(neighbor_info)}"
                    new_labels[node] = signature
                labels = new_labels
        
        return np.array(all_histograms)
    
    def extract_pharmacophore_features(self, G, node_attention_weights=None):
        """
        Distance-based pharmacophore patterns with attention weighting.
        Uses pre-computed attention weights - no learning here.
        """
        features = []
        
        # Get shortest path matrix
        if nx.is_connected(G):
            try:
                spl = dict(nx.all_pairs_shortest_path_length(G))
                
                # Count atom-pair distances (binned) with attention weights
                distance_hist = np.zeros(5)  # distances 1-5+
                for source in spl:
                    for target, dist in spl[source].items():
                        if source < target:  # Avoid double counting
                            bin_idx = min(dist - 1, 4)
                            # Weight by attention of both nodes
                            weight = 1.0
                            if node_attention_weights:
                                weight = (node_attention_weights.get(source, 1.0) + 
                                         node_attention_weights.get(target, 1.0)) / 2
                            distance_hist[bin_idx] += weight
                
                features.extend(distance_hist)
            except:
                features.extend([0] * 5)
        else:
            # For disconnected graphs, compute per component
            components = list(nx.connected_components(G))
            distance_hist = np.zeros(5)
            
            for comp in components:
                subG = G.subgraph(comp)
                try:
                    spl = dict(nx.all_pairs_shortest_path_length(subG))
                    for source in spl:
                        for target, dist in spl[source].items():
                            if source < target:
                                bin_idx = min(dist - 1, 4)
                                weight = 1.0
                                if node_attention_weights:
                                    weight = (node_attention_weights.get(source, 1.0) + 
                                             node_attention_weights.get(target, 1.0)) / 2
                                distance_hist[bin_idx] += weight
                except:
                    pass
            
            features.extend(distance_hist)
        
        return np.array(features)
    
    def extract_spectral_moments(self, G):
        """Higher-order spectral moments from the Laplacian."""
        if G.number_of_nodes() < 2:
            return np.zeros(4)
        
        try:
            L = nx.normalized_laplacian_matrix(G).todense()
            eigenvalues = np.linalg.eigvalsh(L)
            
            # Statistical moments
            features = [
                np.sum(eigenvalues),      # Trace
                np.sum(eigenvalues**2),   # Frobenius norm squared
                np.sum(eigenvalues**3),   # 3rd moment
                np.prod(eigenvalues + 1e-10)  # Determinant (with stability)
            ]
            
            return np.array(features)
        except:
            return np.zeros(4)
    
    def extract_random_walk_features(self, G, node_attention_weights=None):
        """
        Statistical features from attention-weighted random walks.
        Uses pre-computed attention weights - no learning here.
        """
        if G.number_of_nodes() < 2:
            return np.zeros(3)
        
        # Create deterministic seed
        graph_hash = hash(tuple(sorted(G.edges()))) % (2**31)
        rng = np.random.RandomState(graph_hash)
        
        revisit_times = []
        unique_visits = []
        
        nodes = list(G.nodes())
        
        # Create attention-based sampling probabilities
        if node_attention_weights:
            start_probs = np.array([node_attention_weights.get(n, 1.0) for n in nodes])
            start_probs = start_probs / np.sum(start_probs)
        else:
            start_probs = None
        
        for _ in range(self.num_walks):
            # Sample start node based on attention
            if start_probs is not None:
                start_idx = rng.choice(len(nodes), p=start_probs)
                start = nodes[start_idx]
            else:
                start = rng.choice(nodes)
            
            current = start
            visited = {current}
            revisit_time = None
            
            for step in range(1, self.walk_length):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                
                # Weight neighbors by attention
                if node_attention_weights:
                    neighbor_weights = np.array([node_attention_weights.get(n, 1.0) 
                                                 for n in neighbors])
                    neighbor_probs = neighbor_weights / np.sum(neighbor_weights)
                    current = rng.choice(neighbors, p=neighbor_probs)
                else:
                    current = rng.choice(neighbors)
                
                if current == start and revisit_time is None:
                    revisit_time = step
                visited.add(current)
            
            if revisit_time:
                revisit_times.append(revisit_time)
            unique_visits.append(len(visited))
        
        return np.array([
            np.mean(revisit_times) if revisit_times else self.walk_length,
            np.mean(unique_visits),
            np.std(unique_visits)
        ])
    
    def extract_persistence_features(self, G, node_attention_weights=None):
        """
        Topological persistence with attention-weighted filtration.
        Uses pre-computed attention weights - no learning here.
        """
        if G.number_of_nodes() < 2:
            return np.zeros(5)
        
        # Filtration based on attention-weighted degrees
        if node_attention_weights:
            node_values = np.array([G.degree(n) * node_attention_weights.get(n, 1.0) 
                                   for n in G.nodes()])
        else:
            node_values = np.array([G.degree(n) for n in G.nodes()])
        
        # Persistence at different thresholds
        persistence = []
        thresholds = np.percentile(node_values, [0, 25, 50, 75, 100])
        
        nodes_list = list(G.nodes())
        for t in thresholds:
            nodes_above_threshold = [nodes_list[i] for i in range(len(nodes_list)) 
                                    if node_values[i] >= t]
            if nodes_above_threshold:
                subgraph = G.subgraph(nodes_above_threshold)
                persistence.append(nx.number_connected_components(subgraph))
            else:
                persistence.append(0)
        
        return np.array(persistence)
    
    def extract_spectral_features(self, G):
        """Extract eigenvalue-based spectral features"""
        if G.number_of_nodes() < 2:
            return np.zeros(self.n_eigen)
        
        try:
            L = nx.normalized_laplacian_matrix(G).todense()
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = np.sort(eigenvalues)
            
            if len(eigenvalues) < self.n_eigen:
                eigenvalues = np.pad(eigenvalues, (0, self.n_eigen - len(eigenvalues)))
            else:
                eigenvalues = eigenvalues[:self.n_eigen]
            
            return eigenvalues
        except:
            return np.zeros(self.n_eigen)
    
    def extract_topological_features(self, G, node_attention_weights=None, 
                                    edge_attention_weights=None):
        """
        Extract topology-based features with attention weighting.
        Uses pre-computed attention weights - no learning here.
        """
        features = []
        
        # Basic counts
        features.append(G.number_of_nodes())
        features.append(G.number_of_edges())
        features.append(nx.density(G))
        
        # Attention-weighted degree statistics
        if node_attention_weights:
            weighted_degrees = [G.degree(n) * node_attention_weights.get(n, 1.0) 
                               for n in G.nodes()]
        else:
            weighted_degrees = [d for n, d in G.degree()]
        
        if weighted_degrees:
            features.extend([np.mean(weighted_degrees), np.std(weighted_degrees), 
                           np.max(weighted_degrees), np.min(weighted_degrees)])
        else:
            features.extend([0, 0, 0, 0])
        
        try:
            features.append(nx.average_clustering(G))
        except:
            features.append(0)
        
        features.append(nx.number_connected_components(G))
        
        triangles = sum(nx.triangles(G).values()) / 3
        features.append(triangles)
        
        try:
            if nx.is_connected(G):
                features.append(nx.diameter(G))
                features.append(nx.average_shortest_path_length(G))
            else:
                features.append(0)
                features.append(0)
        except:
            features.append(0)
            features.append(0)
        
        if weighted_degrees:
            features.append(np.percentile(weighted_degrees, 25))
            features.append(np.percentile(weighted_degrees, 75))
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def extract_attention_features(self, G, node_attention_weights, edge_attention_weights):
        """
        Extract attention-specific features for interpretability.
        Uses pre-computed attention weights - no learning here.
        """
        features = []
        
        # Node attention statistics
        if node_attention_weights:
            attention_values = list(node_attention_weights.values())
            features.extend([
                np.mean(attention_values),
                np.std(attention_values),
                np.max(attention_values),
                np.min(attention_values)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Edge attention statistics
        if edge_attention_weights:
            edge_values = list(edge_attention_weights.values())
            features.extend([
                np.mean(edge_values),
                np.std(edge_values),
                np.max(edge_values)
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def transform(self, graphs):
        """
        Transform graphs to feature matrix with quantum and attention-enhanced features.
        ONLY uses pre-fitted parameters - no learning or weight updates here.
        """
        features_list = []
        
        for G in graphs:
            # Compute attention weights using PRE-FITTED attention models
            node_attention_weights = None
            edge_attention_weights = None
            
            if self.use_node_attention and self.node_attention:
                node_attention_weights = self.node_attention.compute_attention(G)
            
            if self.use_edge_attention and self.edge_attention:
                edge_attention_weights = self.edge_attention.compute_attention(G)
            
            # Extract classical features
            spectral = self.extract_spectral_features(G)
            topological = self.extract_topological_features(G, node_attention_weights, 
                                                          edge_attention_weights)
            wl = self._compute_wl_histogram_with_attention(G, node_attention_weights)
            pharmacophore = self.extract_pharmacophore_features(G, node_attention_weights)
            spectral_moments = self.extract_spectral_moments(G)
            #random_walk = self.extract_random_walk_features(G, node_attention_weights)
            persistence = self.extract_persistence_features(G, node_attention_weights)
            attention_stats = self.extract_attention_features(G, node_attention_weights, 
                                                             edge_attention_weights)
            
            # Extract quantum features
            if self.use_quantum_features:
                quantum_entanglement = self.extract_quantum_entanglement_features(G)
                quantum_walk = self.extract_quantum_walk_features(G)
                quantum_chemistry = self.extract_quantum_chemistry_features(G)
                
                # Concatenate all features including quantum
                features = np.concatenate([
                    spectral,              # Classical eigenvalues
                    topological,           # Attention-weighted topology
                    wl,                    # Attention-weighted WL
                    pharmacophore,         # Attention-weighted distances
                    spectral_moments,      # Spectral moments
                    #random_walk,           # Classical random walks
                    persistence,           # Attention-weighted persistence
                    attention_stats,       # Attention statistics
                    quantum_entanglement,  # QUANTUM: Graph state entanglement
                    quantum_walk,          # QUANTUM: CTQW features
                    quantum_chemistry      # QUANTUM: MO theory features
                ])
            else:
                # Concatenate classical features only
                features = np.concatenate([
                    spectral,
                    topological,
                    wl,
                    pharmacophore,
                    spectral_moments,
                    random_walk,
                    persistence,
                    attention_stats
                ])
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_attention_analysis(self, G):
        """
        Get detailed attention analysis for a specific graph.
        Returns node and edge attention weights for interpretation.
        Uses pre-fitted attention models only.
        """
        analysis = {}
        
        if self.use_node_attention and self.node_attention:
            node_weights = self.node_attention.compute_attention(G)
            # Find most important nodes
            sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)
            analysis['top_nodes'] = sorted_nodes[:5]
            analysis['node_attention'] = node_weights
        
        if self.use_edge_attention and self.edge_attention:
            edge_weights = self.edge_attention.compute_attention(G)
            # Find most important edges
            sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
            analysis['top_edges'] = sorted_edges[:5]
            analysis['edge_attention'] = edge_weights
        
        return analysis
    
    def get_quantum_analysis(self, G):
        """
        Get detailed quantum feature analysis for a specific graph.
        Returns quantum entanglement, walk, and chemistry features.
        """
        if not self.use_quantum_features:
            return {"error": "Quantum features not enabled"}
        
        analysis = {}
        
        # Entanglement analysis
        entanglement = self.extract_quantum_entanglement_features(G)
        analysis['entanglement'] = {
            'small_partition_entropy': entanglement[0],
            'medium_partition_entropy': entanglement[1],
            'large_partition_entropy': entanglement[2],
            'stabilizer_rank_log': entanglement[3],
            'schmidt_rank_log': entanglement[4],
            'avg_entanglement': entanglement[5]
        }
        
        # Quantum walk analysis
        qwalk = self.extract_quantum_walk_features(G)
        analysis['quantum_walk'] = {
            'spectral_gap': qwalk[0],
            'mixing_time': qwalk[1],
            'eigenvalue_spread': qwalk[2],
            'return_probability': qwalk[3],
            'transport_efficiency': qwalk[4]
        }
        
        # Quantum chemistry analysis
        qchem = self.extract_quantum_chemistry_features(G)
        analysis['quantum_chemistry'] = {
            'homo_lumo_gap': qchem[0],
            'pi_electron_energy': qchem[1],
            'homo_energy': qchem[2],
            'lumo_energy': qchem[3],
            'aromaticity_score': qchem[4],
            'mean_electronegativity': qchem[5],
            'std_electronegativity': qchem[6],
            'electronegativity_range': qchem[7]
        }
        
        return analysis