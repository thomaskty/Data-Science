import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
import logging
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphStore:
    """
    Manages document knowledge graph using NetworkX
    Can be migrated to Neo4j later
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph with parallel edges
        self.storage_path = settings.GRAPH_STORAGE_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing graph if available
        if self.storage_path.exists():
            self.load_graph()

        logger.info("✓ Graph store initialized")

    def add_node(self, node_id: str, node_type: str, **attributes):
        """
        Add a node to the graph

        Args:
            node_id: Unique node identifier
            node_type: Type of node (document, chunk, media)
            **attributes: Additional node attributes
        """
        self.graph.add_node(
            node_id,
            node_type=node_type,
            **attributes
        )

    def add_edge(
            self,
            source_id: str,
            target_id: str,
            relationship_type: str,
            weight: float = 1.0,
            **attributes
    ):
        """
        Add an edge between two nodes

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            weight: Edge weight
            **attributes: Additional edge attributes
        """
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            weight=weight,
            **attributes
        )

    def build_document_graph(
            self,
            document_id: str,
            chunks: List[Dict[str, Any]],
            media: List[Dict[str, Any]]
    ):
        """
        Build graph for a document with its chunks and media

        Args:
            document_id: Document identifier
            chunks: List of chunk dictionaries
            media: List of media dictionaries
        """
        # Add document node
        self.add_node(document_id, node_type='document')

        # Add chunk nodes and edges
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            self.add_node(
                chunk_id,
                node_type='chunk',
                text=chunk['text_content'][:100],  # Store snippet
                page_number=chunk.get('page_number')
            )

            # Document -> Chunk edge
            self.add_edge(
                document_id,
                chunk_id,
                relationship_type='contains',
                weight=1.0
            )

            # Chunk -> Next Chunk edge (sequential)
            if chunk['chunk_index'] > 0:
                prev_chunk_id = chunks[chunk['chunk_index'] - 1]['chunk_id']
                self.add_edge(
                    prev_chunk_id,
                    chunk_id,
                    relationship_type='follows',
                    weight=1.0
                )

        # Add media nodes and edges
        for media_item in media:
            media_id = media_item['media_id']
            self.add_node(
                media_id,
                node_type='media',
                media_type=media_item['media_type'],
                page_number=media_item.get('page_number')
            )

            # Document -> Media edge
            self.add_edge(
                document_id,
                media_id,
                relationship_type='contains',
                weight=1.0
            )

            # Link media to chunks on same page
            page_num = media_item.get('page_number')
            if page_num:
                for chunk in chunks:
                    if chunk.get('page_number') == page_num:
                        self.add_edge(
                            chunk['chunk_id'],
                            media_id,
                            relationship_type='references',
                            weight=0.8
                        )

        logger.info(f"✓ Built graph for document: {document_id}")

    def get_neighbors(
            self,
            node_id: str,
            relationship_type: Optional[str] = None,
            direction: str = 'out'  # 'out', 'in', or 'both'
    ) -> List[str]:
        """
        Get neighboring nodes

        Args:
            node_id: Node identifier
            relationship_type: Filter by relationship type
            direction: Edge direction to follow

        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []

        if direction == 'out':
            neighbors = self.graph.successors(node_id)
        elif direction == 'in':
            neighbors = self.graph.predecessors(node_id)
        else:  # both
            neighbors = set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id))

        neighbors = list(neighbors)

        # Filter by relationship type if specified
        if relationship_type:
            filtered = []
            for neighbor in neighbors:
                edges = self.graph.get_edge_data(node_id, neighbor)
                if edges:
                    for edge in edges.values():
                        if edge.get('relationship_type') == relationship_type:
                            filtered.append(neighbor)
                            break
            return filtered

        return neighbors

    def get_context_subgraph(
            self,
            node_id: str,
            max_hops: int = 2
    ) -> nx.DiGraph:
        """
        Get subgraph around a node (for context expansion)

        Args:
            node_id: Central node ID
            max_hops: Maximum distance from central node

        Returns:
            NetworkX subgraph
        """
        if node_id not in self.graph:
            return nx.DiGraph()

        # Get nodes within max_hops
        nodes = {node_id}
        for _ in range(max_hops):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.successors(node))
                new_nodes.update(self.graph.predecessors(node))
            nodes.update(new_nodes)

        return self.graph.subgraph(nodes).copy()

    def save_graph(self):
        """Save graph to disk"""
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"✓ Graph saved: {self.storage_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save graph: {e}")

    def load_graph(self):
        """Load graph from disk"""
        try:
            with open(self.storage_path, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"✓ Graph loaded: {self.storage_path}")
        except Exception as e:
            logger.error(f"✗ Failed to load graph: {e}")
            self.graph = nx.MultiDiGraph()

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': self._count_node_types(),
            'relationship_types': self._count_relationship_types()
        }

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        counts = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type"""
        counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relationship_type', 'unknown')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts

    def clear_graph(self):
        """Clear all graph data"""
        self.graph.clear()
        logger.info("✓ Graph cleared")


# Create singleton instance
graph_store = GraphStore()