final public class DAGclipboard {
	private static DAGclipboard instance = null;
	public DAGconn tconn;
	public DAGnode tnode;
	public Boolean isCopiedNode;
	public Boolean isCopied;
	
	private DAGclipboard() {
		isCopiedNode=false;
		isCopied=false;
	}
	public static DAGclipboard getInstance() {
		if(instance == null) {
			instance = new DAGclipboard();
		}
		return instance;
	}
	
	public void setCopiedAsWeight(DAGconn cconn) {
		tconn = cconn.clone();
		isCopiedNode = false;
		isCopied = true;
	}
	
	public void setCopiedAsNode(DAGnode cnode) {
		tnode = cnode.clone();
		isCopiedNode = true;
		isCopied = true;
	}
	
	public DAGnode getCopiedNode() {
		DAGnode ttnode = tnode.clone();
		ttnode.active = true;
		ttnode.selected = true;
		return ttnode;
	}
	
	public DAGconn getCopiedConn() {
		DAGconn ttconn = tconn.clone();
		ttconn.active = true;
		ttconn.selected = true;
		return ttconn;
	}
}
