import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRootPane;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.SpringLayout;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import org.w3c.dom.*;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

public class DAGgraph {
	DAGclipboard clipboard;
    private int nextNodeId;
    private int nextConnId;
    private Vector<DAGnode> nodeArr;
    private Vector<DAGconn> connArr;
    private Hashtable<Integer,Integer> connLookup;
    private Hashtable<Integer,Integer> nodeLookup;

    private BufferedImage staticElements;

    private int mode;
    private int selectedItem;

    private int wLast, hLast;
	private JDialog paramEdit;
	private Container paramContainer;
    private DAGPanel canvas;
    private int defNmsc[], defMsc1[], defMsc2[], defSz[];
    private float defNeta, defNsigma, defWeta, defWsigma;
    private int defMapCt;
    
    static Node getChild(Node parent, String name) {
    	for (Node child = parent.getFirstChild(); child != null; child = child.getNextSibling()) {
    		if (child instanceof Element && name.equals(child.getNodeName())) {
    			return child;
    		}
		}
		return null;
    }
    
	public DAGgraph(final JToggleButton tLaunchButton, DAGPanel tcanvas) throws ParserConfigurationException {
		
		
		
		DocumentBuilderFactory docBuilderFactory = DocumentBuilderFactory.newInstance();
		
		DocumentBuilder docBuilder = docBuilderFactory.newDocumentBuilder();
		Document doc = docBuilder.newDocument();
		File f = new File("defaults.xml");
		if(!f.exists()) {
        	try {
        		FileWriter fstream = new FileWriter("defaults.xml");
        		BufferedWriter out = new BufferedWriter(fstream);
        		
        		Element rootElement = doc.createElement("defaults");
        		
        		doc.appendChild(rootElement);
        		Element nodeRoot = doc.createElement("node");
        		Element nodeEta = doc.createElement("eta");
        		nodeEta.appendChild(doc.createTextNode("1"));
        		Element nodeSigma = doc.createElement("sigma");
        		nodeSigma.appendChild(doc.createTextNode("1"));
        		Element nodeMc = doc.createElement("map_count");
        		nodeMc.appendChild(doc.createTextNode("1"));
        		Element nodeMsc = doc.createElement("multiscale");
        		Element nmscx = doc.createElement("x");
        		Element nmscy = doc.createElement("y");
        		Element nmscz = doc.createElement("z");
        		nmscx.appendChild(doc.createTextNode("1"));
        		nmscy.appendChild(doc.createTextNode("1"));
        		nmscz.appendChild(doc.createTextNode("1"));
        		nodeMsc.appendChild(nmscx);
        		nodeMsc.appendChild(nmscy);
        		nodeMsc.appendChild(nmscz);
        		
        		nodeRoot.appendChild(nodeEta);
        		nodeRoot.appendChild(nodeSigma);
        		nodeRoot.appendChild(nodeMc);
        		nodeRoot.appendChild(nodeMsc);
                rootElement.appendChild(nodeRoot);
                Element weightRoot = doc.createElement("weight");
                
        		Element weightEta = doc.createElement("eta");
        		weightEta.appendChild(doc.createTextNode("1"));
        		Element weightSigma = doc.createElement("sigma");
        		weightSigma.appendChild(doc.createTextNode("1"));
        		Element weightMsc1 = doc.createElement("multiscale_pixel_size");
        		Element weightMsc2 = doc.createElement("multiscale_pixel_space");
        		Element weightSz = doc.createElement("size");
        		Element wmsc1x = doc.createElement("x");
        		Element wmsc1y = doc.createElement("y");
        		Element wmsc1z = doc.createElement("z");
        		wmsc1x.appendChild(doc.createTextNode("1"));
        		wmsc1y.appendChild(doc.createTextNode("1"));
        		wmsc1z.appendChild(doc.createTextNode("1"));
        		Element wmsc2x = doc.createElement("x");
        		Element wmsc2y = doc.createElement("y");
        		Element wmsc2z = doc.createElement("z");
        		wmsc2x.appendChild(doc.createTextNode("1"));
        		wmsc2y.appendChild(doc.createTextNode("1"));
        		wmsc2z.appendChild(doc.createTextNode("1"));
        		Element wszx = doc.createElement("x");
        		Element wszy = doc.createElement("y");
        		Element wszz = doc.createElement("z");
        		wszx.appendChild(doc.createTextNode("1"));
        		wszy.appendChild(doc.createTextNode("1"));
        		wszz.appendChild(doc.createTextNode("1"));
        		weightMsc1.appendChild(wmsc1x);
        		weightMsc1.appendChild(wmsc1y);
        		weightMsc1.appendChild(wmsc1z);
        		weightMsc2.appendChild(wmsc2x);
        		weightMsc2.appendChild(wmsc2y);
        		weightMsc2.appendChild(wmsc2z);
        		weightSz.appendChild(wszx);
        		weightSz.appendChild(wszy);
        		weightSz.appendChild(wszz);
        		
        		weightRoot.appendChild(weightEta);
        		weightRoot.appendChild(weightSigma);
        		weightRoot.appendChild(weightSz);
        		weightRoot.appendChild(weightMsc1);
        		weightRoot.appendChild(weightMsc2);
                rootElement.appendChild(weightRoot);
                
                
                TransformerFactory transformerFactory = TransformerFactory.newInstance();
                Transformer transformer = transformerFactory.newTransformer();
                
                DOMSource source = new DOMSource(doc);
                StreamResult result =  new StreamResult(out);
                transformer.setOutputProperty(OutputKeys.INDENT, "yes");
                transformer.transform(source, result);
                out.close();
        	} catch(Exception e1) {
        		System.out.println("All sorts of messed up");
        	}
        }
        
		try {

            // normalize text representation
			doc = docBuilder.parse (new File("defaults.xml"));
            doc.getDocumentElement().normalize();
            if(!doc.getDocumentElement().getNodeName().equalsIgnoreCase("defaults"))
            	throw(new Throwable("Invalid XML structure"));
            Element docEle = doc.getDocumentElement();
            Node xmlNode = docEle.getElementsByTagName("node").item(0);
            Node xmlWeight = doc.getElementsByTagName("weight").item(0);
            defNeta = Float.parseFloat( getChild(xmlNode,"eta").getTextContent() );
    		defWeta = Float.parseFloat( getChild(xmlWeight,"eta").getTextContent() );
    		defNsigma = Float.parseFloat( getChild(xmlNode,"sigma").getTextContent() );
    		defWsigma = Float.parseFloat( getChild(xmlWeight,"sigma").getTextContent() );
    		defMapCt = Integer.parseInt( getChild(xmlNode,"map_count").getTextContent() );
    		defSz = new int[3];
    		defNmsc = new int[3];
    		defMsc1 = new int[3];
    		defMsc2 = new int[3];
    		defSz[0] = Integer.parseInt(getChild(getChild(xmlWeight,"size"),"y").getTextContent());
    		defSz[1] = Integer.parseInt(getChild(getChild(xmlWeight,"size"),"x").getTextContent());
    		defSz[2] = Integer.parseInt(getChild(getChild(xmlWeight,"size"),"z").getTextContent());
    		defNmsc[0] = Integer.parseInt(getChild(getChild(xmlNode,"multiscale"),"y").getTextContent());
    		defNmsc[1] = Integer.parseInt(getChild(getChild(xmlNode,"multiscale"),"x").getTextContent());
    		defNmsc[2] = Integer.parseInt(getChild(getChild(xmlNode,"multiscale"),"z").getTextContent());
    		defMsc1[0] = Integer.parseInt(getChild(getChild(xmlWeight,"multiscale_pixel_space"),"y").getTextContent());
    		defMsc1[1] = Integer.parseInt(getChild(getChild(xmlWeight,"multiscale_pixel_space"),"x").getTextContent());
    		defMsc1[2] = Integer.parseInt(getChild(getChild(xmlWeight,"multiscale_pixel_space"),"z").getTextContent());
    		defMsc2[0] = Integer.parseInt(getChild(getChild(xmlWeight,"multiscale_pixel_size"),"y").getTextContent());
    		defMsc2[1] = Integer.parseInt(getChild(getChild(xmlWeight,"multiscale_pixel_size"),"x").getTextContent());
    		defMsc2[2] = Integer.parseInt(getChild(getChild(xmlWeight,"multiscale_pixel_size"),"z").getTextContent());

        }catch (Throwable t) {
        	t.printStackTrace();
        }
		
		
		canvas = tcanvas;

    	clipboard = DAGclipboard.getInstance();
		paramEdit = new JDialog();
		paramEdit.addWindowListener(new WindowAdapter() {
				public void windowClosing(WindowEvent e) {
					tLaunchButton.setSelected(false);
				}
			});
		paramEdit.setAlwaysOnTop(true);
		JRootPane root = paramEdit.getRootPane();
		root.putClientProperty("Window.style", "small");
		
		paramContainer = paramEdit.getContentPane();
		SpringLayout layout = new SpringLayout();
		paramContainer.setLayout(layout);
		nodeArr = new Vector<DAGnode>();
		connArr = new Vector<DAGconn>();
		nodeLookup = new Hashtable<Integer, Integer>();
		connLookup = new Hashtable<Integer, Integer>();
		wLast=1;
		hLast=1;
		nextNodeId = 0;
		nextConnId = 0;

		this.modeSet(0); // nothing mode
    }
    
    public void selfOrganize(boolean updateView) {
    	boolean explored[] = new boolean[nodeArr.size()];
    	boolean exploredAlt[] = new boolean[nodeArr.size()];
    	ArrayList<ArrayList<Integer>> nodeClusters = new ArrayList<ArrayList<Integer>>(0);
    	for(int i=0; i<nodeArr.size(); i++) {
    		explored[i]=exploredAlt[i]=false;
    	}
    	int clusteredCt = 1;
    	int clusterId = 0;
    	while(clusteredCt>0) {
    		nodeClusters.add(new ArrayList<Integer>(0));
    		clusteredCt = 0;
    		for(int i=0; i<nodeArr.size(); i++) {
    			if(!explored[i]) {
    				DAGnode tnode = nodeArr.get(i);
    				boolean aok = true;
    				for(int j=0; j<tnode.connDest.size(); j++) {
    					DAGconn tconn = connArr.get(connLookup.get(tnode.connDest.get(j)));
    					int ti = nodeLookup.get(tconn.from);
    					aok &= explored[ti];
    				}
    				if(aok) {
						nodeClusters.get(clusterId).add(tnode.id);
						clusteredCt++;
						exploredAlt[i] = true;
					}
    			}
    		}
    		System.arraycopy(exploredAlt,0,explored,0,nodeArr.size());
    		clusterId++;
    	}
    	nodeClusters.remove(clusterId-1);
    	// Update everything
    	if(updateView) {
    		selfLayout(nodeClusters);
    	}
    	
    	Vector<DAGnode> nodeArrPrime = new Vector<DAGnode>();
        Hashtable<Integer,Integer> nodeLookupPrime = new Hashtable<Integer,Integer>();
        
        for(int i=0; i<nodeClusters.size(); i++) {
        	for(int j=0; j<nodeClusters.get(i).size(); j++) {
        		DAGnode tnode = nodeArr.get(nodeLookup.get( nodeClusters.get(i).get(j) ));
        		nodeArrPrime.add( tnode );
        		nodeLookupPrime.put(tnode.id, nodeArrPrime.size()-1);
        	}
        }
        
        nodeArr = nodeArrPrime;
        nodeLookup = nodeLookupPrime;
    }
    
    public void selfLayout(ArrayList<ArrayList<Integer>> clusters) {
    	for(int i=0; i<connArr.size(); i++) {
    		connArr.get(i).centerPushx=0;
    		connArr.get(i).centerPushy=0;
    	}
    	for(int i=0; i<clusters.size(); i++) {
    		int maxNodeWidth = 0;
    		for(int j=0; j<clusters.get(i).size(); j++) {
    			int thisWidth = nodeArr.get(nodeLookup.get(clusters.get(i).get(j))).width;
    			if(thisWidth > maxNodeWidth) {
    				maxNodeWidth = thisWidth;
    			}
    		}
    		for(int j=0; j<clusters.get(i).size(); j++) {
    			Dimension md = canvas.getSize();
    			int cellheight = (md.height-40)/(clusters.size());
    			int cellwidth = (md.width - maxNodeWidth - 4)/(clusters.get(i).size());
    			int ty = (md.height - 10 - i*cellheight - cellheight/2);
    			int tx = (j*cellwidth + 2 + maxNodeWidth/2 + cellwidth/2);
    			int tid = clusters.get(i).get(j);
    			nodeArr.get(nodeLookup.get(tid)).setPos(tx,ty);
	    		for(int k=0; k<nodeArr.get(nodeLookup.get(tid)).connDest.size(); k++) {
                	connArr.get(connLookup.get( nodeArr.get(nodeLookup.get(tid)).connDest.get(k) )).setEnd(tx,ty+6);
            	}
            	for(int k=0; k<nodeArr.get(nodeLookup.get(tid)).connOrigin.size(); k++) {
					connArr.get(connLookup.get( nodeArr.get(nodeLookup.get(tid)).connOrigin.get(k) )).setStart(tx,ty-13);
            	}
    		}
    	}
    	redrawBackground(wLast, hLast);
		canvas.repaint();
    }
    
    public void addSensToNode(int whichNode) {
    	nodeArr.get(nodeLookup.get(whichNode)).withSens = true;
    }
    
    public void addErrToNode(int whichNode) {
    	nodeArr.get(nodeLookup.get(whichNode)).computeError = true;
    }
    
    public int numUncomputed() {
    	int ct=0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()==0) {
    			ct++;
    		}
    	}
    	return ct;
    }
    
    public int numComputed() {
    	int ct=0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			ct++;
    		}
    	}
    	return ct;
    }
    
    public int numConn() {
    	return connArr.size();
    }
    
    public char[] getUncomputedName(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()==0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).name.toCharArray();
    			}
    			tmpId++;
    		}
    	}
    	char t[] = {'?'};
    	return t;
    }
    
    public char[] getConnName(int whichConn) {
    	return connArr.get(whichConn).name.toCharArray();
    }
    
    public char[] getComputedName(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).name.toCharArray();
    			}
    			tmpId++;
    		}
    	}
    	char t[] = {'?'};
    	return t;
    }
    
    public int getUncomputedMapCt(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()==0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).mapCt;
    			}
    			tmpId++;
    		}
    	}
    	return 0;
    }
    
    public int getComputedMapCt(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).mapCt;
    			}
    			tmpId++;
    		}
    	}
    	return 0;
    }
    
    public int getUncomputedId(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()==0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).id;
    			}
    			tmpId++;
    		}
    	}
    	return 0;
    }
    
    public int getComputedId(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).id;
    			}
    			tmpId++;
    		}
    	}
    	return 0;
    }
    
    public boolean computedHasSens(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).withSens;
    			}
    			tmpId++;
    		}
    	}
    	return false;
    }
    
    public boolean computedHasErr(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).computeError;
    			}
    			tmpId++;
    		}
    	}
    	return false;
    }
    
    public int[] getSpace(int whichNode, boolean computed) {
    	int spacing[] = new int[3];
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if((!computed && nodeArr.get(i).connDest.size()==0) || (computed && nodeArr.get(i).connDest.size()>0)) {
    			if(tmpId == whichNode) {
    				for(int j=0; j<3; j++)
    					spacing[j] = nodeArr.get(i).msc[j];
    				break;
    			}
    			tmpId++;
    		}
    	}
    	return spacing;
    }
    
    public int[] getConnSpace(int whichConn) {
    	return connArr.get(whichConn).msc1;
    }
    
    public int[] getConnBlock(int whichConn) {
    	int block[] = new int[3];
    	for(int i=0; i<3; i++)
    		block[i] = connArr.get(whichConn).msc2[i];
    	return block;
    }
    
    public int[] getConnSize(int whichConn) {
    	int tmpSize[] = new int[5];
    	tmpSize[0] = connArr.get(whichConn).mict;
    	for(int i=0; i<3; i++)
    		tmpSize[i+1] = connArr.get(whichConn).sz[i];
    	tmpSize[4] = connArr.get(whichConn).moct;
    	return tmpSize;
    }
    
    public float[] getConnVals(int whichConn) {
    	int sz[] = connArr.get(whichConn).sz;
    	int mict=connArr.get(whichConn).mict;
		int moct=connArr.get(whichConn).moct;
    	float tmpW[] = new float[mict*moct*sz[0]*sz[1]*sz[2]];
    	for(int mi=0; mi<mict; mi++) {
    		for(int x=0; x<sz[0]; x++) {
    			for(int y=0; y<sz[1]; y++) {
    				for(int z=0; z<sz[2]; z++) {
    					for(int mo=0; mo<moct; mo++) {
    						tmpW[mi + x*mict + y*mict*sz[0] + z*mict*sz[0]*sz[1] + mo*mict*sz[0]*sz[1]*sz[2]] = connArr.get(whichConn).data[x + y*sz[0] + z*sz[0]*sz[1] + mi*sz[0]*sz[1]*sz[2] + mo*sz[0]*sz[1]*sz[2]*mict];
    					}
    				}
    			}
    		}
    	}
    	return tmpW;
    }
    
    public float getConnEta(int whichConn) {
    	return connArr.get(whichConn).eta;
    }
    
    public int getConnTo(int whichConn) {
    	return connArr.get(whichConn).to;
    }
    
    public int getConnFrom(int whichConn) {
    	return connArr.get(whichConn).from;
    }
    
    public float[] getComputedBias(int whichNode) {
    	float tmpBias[] = new float[1];
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				tmpBias = new float[nodeArr.get(i).mapCt];
    				System.arraycopy(nodeArr.get(i).bias, 0, tmpBias, 0, nodeArr.get(i).mapCt);
    				break;
    			}
    			tmpId++;
    		}
    	}
    	return tmpBias;
    }
    
    public float getComputedEta(int whichNode) {
    	int tmpId = 0;
    	for(int i=0; i<nodeArr.size(); i++) {
    		if(nodeArr.get(i).connDest.size()>0) {
    			if(tmpId == whichNode) {
    				return nodeArr.get(i).eta;
    			}
    			tmpId++;
    		}
    	}
    	return 0;
    }
    
    public void copySelected() {
    	if(mode==3) {
    		clipboard.setCopiedAsNode(nodeArr.get((int)nodeLookup.get(selectedItem)));
    	} else if(mode==4) {
    		clipboard.setCopiedAsWeight(connArr.get((int)connLookup.get(selectedItem)));
    	}
    }
    
    public void cutSelected() {
    	if(mode==3) {
    		clipboard.setCopiedAsNode(nodeArr.get((int)nodeLookup.get(selectedItem)));
    		removeNode(selectedItem);
    	} else if(mode==4) {
    		clipboard.setCopiedAsWeight(connArr.get((int)connLookup.get(selectedItem)));
    		removeConn(selectedItem);
    	}
    	redrawBackground(wLast, hLast);
		canvas.repaint();
    }
    
    public void pasteSelected() {
    	if(clipboard.isCopied) {
    		if(clipboard.isCopiedNode) {
    			addNode(clipboard.getCopiedNode());
    		} else {
    			addConn(clipboard.getCopiedConn());
    		}
    	}
    }
    
	public boolean checkForLoops(Vector<Integer> visitedConns, int nextNode, int originNode) {
		if(nextNode == originNode) {
			return true;
		}
		Vector<Integer> nextConns = nodeArr.get((int)nodeLookup.get(nextNode)).connOrigin;
		for(int i=0; i<nextConns.size(); i++) {
			if(visitedConns.contains((Integer)nextConns.get(i))) {
				return true;
			}
			visitedConns.add((Integer)nextConns.get(i));
			int nextNode2 = connArr.get((int)connLookup.get(nextConns.get(i))).to;
			if(this.checkForLoops(visitedConns,nextNode2,originNode)) {
				return true;
			}
		}
		return false;
	}

	public void showParamEdit(Boolean showit) {
		paramEdit.setVisible(showit);
	}

	public void modeSet(int newmode) {
		mode = newmode;
		switch(mode) {
			case 5:
			case 3:
				// node selected
				JPanel spacingPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField spEdit[] = new JTextField[3];
				for(int i=0; i<3; i++) {
					spEdit[i] = new JTextField(Integer.toString(nodeArr.get(nodeLookup.get(selectedItem)).msc[i]),2);
					spacingPanel.add(spEdit[i]);
				}
				
				JPanel fmPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField fmctEdit = new JTextField(Integer.toString(nodeArr.get((int)nodeLookup.get(selectedItem)).mapCt),4);
				JButton fmStore = new JButton("Confirm");
				fmPanel.add(fmctEdit);
				fmPanel.add(fmStore);
				
				paramContainer.removeAll();
				
				String[] labels = {"Name: ","Computed: ","Sigma: ","Map Count: ","Spacing: ","+Sens: ","+Error: ","Eta: ",""};
				int numPairs = labels.length;
				
				final JTextField nNameEdit = new JTextField(nodeArr.get((int)nodeLookup.get(selectedItem)).name,14);
				JCheckBox computedEdit = new JCheckBox("",nodeArr.get((int)nodeLookup.get(selectedItem)).connDest.size()!=0);
				computedEdit.setEnabled(false);
				final JTextField initSigEdit = new JTextField(Float.toString(nodeArr.get(nodeLookup.get(selectedItem)).sigma),4);
				final JCheckBox hasSensEdit = new JCheckBox("",nodeArr.get((int)nodeLookup.get(selectedItem)).withSens);
				hasSensEdit.setEnabled(nodeArr.get((int)nodeLookup.get(selectedItem)).connOrigin.size()!=0);
				final JCheckBox errEdit = new JCheckBox("",nodeArr.get((int)nodeLookup.get(selectedItem)).computeError);
				errEdit.setEnabled(nodeArr.get((int)nodeLookup.get(selectedItem)).connOrigin.size()==0);
				final JTextField etaEdit = new JTextField(Float.toString(nodeArr.get(nodeLookup.get(selectedItem)).eta));
				JButton reInit = new JButton("Reinitialize Bias");
				
				JComponent[] editables = {nNameEdit,computedEdit,initSigEdit,fmPanel,spacingPanel,hasSensEdit,errEdit,etaEdit,reInit};
				
				for(int i=0; i<numPairs; i++) {
					JLabel l = new JLabel(labels[i], JLabel.TRAILING);
					paramContainer.add(l);
					l.setLabelFor(editables[i]);
					paramContainer.add(editables[i]);
				}
				
				SpringUtilities.makeCompactGrid(paramContainer,
				                                numPairs, 2,
				                                0, 0,
				                                0, 0);
				
				fmctEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						changeMapCt(selectedItem,Integer.parseInt(fmctEdit.getText()),Float.parseFloat(initSigEdit.getText()));
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				
				fmStore.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						changeMapCt(selectedItem,Integer.parseInt(fmctEdit.getText()),Float.parseFloat(initSigEdit.getText()));
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				
				reInit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						nodeArr.get(nodeLookup.get(selectedItem)).reInitialize();
					}
				});
				
				hasSensEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						if(hasSensEdit.isSelected()) {
							enableSensUpstream(selectedItem);
						} else {
							disableSensDownstream(selectedItem);
						}
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				
				nNameEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						nodeArr.get((int)nodeLookup.get(selectedItem)).renameNode(nNameEdit.getText());
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				
				etaEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						nodeArr.get((int)nodeLookup.get(selectedItem)).eta = Float.parseFloat(etaEdit.getText());
					}
				});
				
				errEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						nodeArr.get((int)nodeLookup.get(selectedItem)).computeError = errEdit.isSelected();
						if(!errEdit.isSelected()) {
							disableSensDownstream(selectedItem);
						}
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				
				spEdit[0].getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							nodeArr.get((int)nodeLookup.get(selectedItem)).msc[0] = Integer.parseInt(spEdit[0].getText());
						} catch(Exception e) {}
					}
				});
				
				spEdit[1].getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							nodeArr.get((int)nodeLookup.get(selectedItem)).msc[1] = Integer.parseInt(spEdit[1].getText());
						} catch(Exception e) {}
					}
				});
				
				spEdit[2].getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							nodeArr.get((int)nodeLookup.get(selectedItem)).msc[2] = Integer.parseInt(spEdit[2].getText());
						} catch(Exception e) {}
					}
				});
				
				initSigEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							nodeArr.get((int)nodeLookup.get(selectedItem)).sigma = Float.parseFloat(initSigEdit.getText());
						} catch(Exception e) {}
					}
				});
					
				break;
			case 4:
			case 9:
				// weight selected
				
				paramContainer.removeAll();
				
				final JTextField wNameEdit = new JTextField(connArr.get((int)connLookup.get(selectedItem)).name,14);
				final JTextField wInitSigEdit = new JTextField(Float.toString(connArr.get(connLookup.get(selectedItem)).sigma),4);
				
				JPanel mictEdit = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				JPanel moctEdit = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				
				final JTextField mictValEdit = new JTextField(Integer.toString(connArr.get(connLookup.get(selectedItem)).mict),4);
				final JTextField moctValEdit = new JTextField(Integer.toString(connArr.get(connLookup.get(selectedItem)).moct),4);
				mictEdit.add(mictValEdit);
				JButton mictConfirm = new JButton("Confirm");
				mictEdit.add(mictConfirm);
				moctEdit.add(moctValEdit);
				JButton moctConfirm = new JButton("Confirm");
				moctEdit.add(moctConfirm);
				
				moctValEdit.setEnabled(connArr.get(connLookup.get(selectedItem)).to==-1);
				moctConfirm.setEnabled(connArr.get(connLookup.get(selectedItem)).to==-1);
				
				mictValEdit.setEnabled(connArr.get(connLookup.get(selectedItem)).from==-1);
				mictConfirm.setEnabled(connArr.get(connLookup.get(selectedItem)).from==-1);
				
				JPanel szPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField szxEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).sz[0]),2);
				final JTextField szyEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).sz[1]),2);
				final JTextField szzEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).sz[2]),2);
				JButton szConfirm = new JButton("Confirm");
				szPanel.add(szyEdit);
				szPanel.add(szxEdit);
				szPanel.add(szzEdit);
				szPanel.add(szConfirm);
				
				JPanel msc1Panel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField msc1xEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).msc1[0]),2);
				final JTextField msc1yEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).msc1[1]),2);
				final JTextField msc1zEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).msc1[2]),2);
				msc1Panel.add(msc1yEdit);
				msc1Panel.add(msc1xEdit);
				msc1Panel.add(msc1zEdit);
				
				JPanel msc2Panel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField msc2xEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).msc2[0]),2);
				final JTextField msc2yEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).msc2[1]),2);
				final JTextField msc2zEdit = new JTextField(Integer.toString(connArr.get((int)connLookup.get(selectedItem)).msc2[2]),2);
				msc2Panel.add(msc2yEdit);
				msc2Panel.add(msc2xEdit);
				msc2Panel.add(msc2zEdit);
				
				final JTextField wetaEdit = new JTextField(Float.toString(connArr.get((int)connLookup.get(selectedItem)).eta));
				JButton wReInit = new JButton("Reinitialize Weight");
				
				String[] wlabels = {"Name: ","Sigma: ","In Map Ct: ","Out Map Ct: ","Size: ","Spacing: ","Block size: ","Eta: ",""};
				int wnumPairs = wlabels.length;
				JComponent wEditables[] = {wNameEdit, wInitSigEdit, mictEdit, moctEdit, szPanel, msc1Panel, msc2Panel, wetaEdit, wReInit};
				
				for(int i=0; i<wnumPairs; i++) {
					JLabel l = new JLabel(wlabels[i], JLabel.TRAILING);
					paramContainer.add(l);
					l.setLabelFor(wEditables[i]);
					paramContainer.add(wEditables[i]);
				}
				
				SpringUtilities.makeCompactGrid(paramContainer,
				                                wnumPairs, 2,
				                                0, 0,
				                                0, 0);
				
				wNameEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get((int)connLookup.get(selectedItem)).name = wNameEdit.getText();
					}
				});
				wetaEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get((int)connLookup.get(selectedItem)).eta = Float.parseFloat(wetaEdit.getText());
					}
				});
				
				szxEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(-1,
								Integer.parseInt(szxEdit.getText()),
								Integer.parseInt(szyEdit.getText()),
								Integer.parseInt(szzEdit.getText()),
								-1);
					}
				});
				szyEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(-1,
								Integer.parseInt(szxEdit.getText()),
								Integer.parseInt(szyEdit.getText()),
								Integer.parseInt(szzEdit.getText()),
								-1);
					}
				});
				szzEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(-1,
								Integer.parseInt(szxEdit.getText()),
								Integer.parseInt(szyEdit.getText()),
								Integer.parseInt(szzEdit.getText()),
								-1);
					}
				});
				szConfirm.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(-1,
								Integer.parseInt(szxEdit.getText()),
								Integer.parseInt(szyEdit.getText()),
								Integer.parseInt(szzEdit.getText()),
								-1);
					}
				});
				moctValEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(-1,
								-1,
								-1,
								-1,
								Integer.parseInt(moctValEdit.getText()));
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				moctConfirm.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(-1,
								-1,
								-1,
								-1,
								Integer.parseInt(moctValEdit.getText()));
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				mictValEdit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(Integer.parseInt(mictValEdit.getText()),
								-1,
								-1,
								-1,
								-1);
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				mictConfirm.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).changeSize(Integer.parseInt(mictValEdit.getText()),
								-1,
								-1,
								-1,
								-1);
						redrawBackground(wLast, hLast);
						canvas.repaint();
					}
				});
				
				wReInit.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						connArr.get(connLookup.get(selectedItem)).reInitialize();
					}
				});
				
				wInitSigEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get((int)connLookup.get(selectedItem)).sigma = Float.parseFloat(wInitSigEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc1xEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get(connLookup.get(selectedItem)).msc1[0] = Integer.parseInt(msc1xEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc1yEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get(connLookup.get(selectedItem)).msc1[1] = Integer.parseInt(msc1yEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc1zEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get(connLookup.get(selectedItem)).msc1[2] = Integer.parseInt(msc1zEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc2xEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get(connLookup.get(selectedItem)).msc2[0] = Integer.parseInt(msc2xEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc2yEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get(connLookup.get(selectedItem)).msc2[1] = Integer.parseInt(msc2yEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc2zEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							connArr.get(connLookup.get(selectedItem)).msc2[2] = Integer.parseInt(msc2zEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				
				break;
			default:
				// Modify defaults
				paramContainer.removeAll();
				
				final JTextField defnSigEdit = new JTextField(Float.toString(defNsigma),4);
				final JTextField defnEtaEdit = new JTextField(Float.toString(defNeta),4);
				final JTextField defnMapCtEdit = new JTextField(Integer.toString(defMapCt),4);

				JPanel nmscDefPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField nmscxDefEdit = new JTextField(Integer.toString(defNmsc[0]),2);
				final JTextField nmscyDefEdit = new JTextField(Integer.toString(defNmsc[1]),2);
				final JTextField nmsczDefEdit = new JTextField(Integer.toString(defNmsc[2]),2);
				nmscDefPanel.add(nmscyDefEdit);
				nmscDefPanel.add(nmscxDefEdit);
				nmscDefPanel.add(nmsczDefEdit);
				
				final JTextField defwSigEdit = new JTextField(Float.toString(defWsigma),4);
				final JTextField defwEtaEdit = new JTextField(Float.toString(defWeta),4);

				JPanel szDefPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField szxDefEdit = new JTextField(Integer.toString(defSz[0]),2);
				final JTextField szyDefEdit = new JTextField(Integer.toString(defSz[1]),2);
				final JTextField szzDefEdit = new JTextField(Integer.toString(defSz[2]),2);
				szDefPanel.add(szyDefEdit);
				szDefPanel.add(szxDefEdit);
				szDefPanel.add(szzDefEdit);
				
				JPanel msc1DefPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField msc1xDefEdit = new JTextField(Integer.toString(defMsc1[0]),2);
				final JTextField msc1yDefEdit = new JTextField(Integer.toString(defMsc1[1]),2);
				final JTextField msc1zDefEdit = new JTextField(Integer.toString(defMsc1[2]),2);
				msc1DefPanel.add(msc1yDefEdit);
				msc1DefPanel.add(msc1xDefEdit);
				msc1DefPanel.add(msc1zDefEdit);
				
				JPanel msc2DefPanel = new JPanel(new FlowLayout(FlowLayout.LEFT,0,0));
				final JTextField msc2xDefEdit = new JTextField(Integer.toString(defMsc2[0]),2);
				final JTextField msc2yDefEdit = new JTextField(Integer.toString(defMsc2[1]),2);
				final JTextField msc2zDefEdit = new JTextField(Integer.toString(defMsc2[2]),2);
				msc2DefPanel.add(msc2yDefEdit);
				msc2DefPanel.add(msc2xDefEdit);
				msc2DefPanel.add(msc2zDefEdit);
				
				JPanel wdefs = new JPanel(new SpringLayout());
				
				String[] wdefLabels = {"Size: ","Spacing: ","Block size: ","Eta: ","Sigma: "};
				int wdefNumPairs = wdefLabels.length;
				JComponent defEditables[] = {szDefPanel, msc1DefPanel, msc2DefPanel, defwEtaEdit, defwSigEdit};
				
				
				for(int i=0; i<wdefNumPairs; i++) {
					JLabel l = new JLabel(wdefLabels[i], JLabel.TRAILING);
					wdefs.add(l);
					l.setLabelFor(defEditables[i]);
					wdefs.add(defEditables[i]);
				}
				
				SpringUtilities.makeCompactGrid(wdefs,
                        wdefNumPairs, 2,
                        0, 0,
                        0, 0);
				
				JPanel ndefs = new JPanel(new SpringLayout());
				
				String[] ndefLabels = {"Map Count: ","Spacing: ","Eta: ","Sigma: "};
				int ndefNumPairs = ndefLabels.length;
				JComponent ndefEditables[] = {defnMapCtEdit, nmscDefPanel, defnEtaEdit, defnSigEdit};
				
				
				for(int i=0; i<ndefNumPairs; i++) {
					JLabel l = new JLabel(ndefLabels[i], JLabel.TRAILING);
					ndefs.add(l);
					l.setLabelFor(ndefEditables[i]);
					ndefs.add(ndefEditables[i]);
				}
				
				SpringUtilities.makeCompactGrid(ndefs,
                        ndefNumPairs, 2,
                        0, 0,
                        0, 0);
				
				wdefs.setBorder(BorderFactory.createTitledBorder("Weights"));
				ndefs.setBorder(BorderFactory.createTitledBorder("Nodes"));
				
				paramContainer.add(ndefs);
				paramContainer.add(wdefs);
				SpringUtilities.makeCompactGrid(paramContainer,
										2,1,
										0,0,
										0,0);
				
				// Add callbacks
				msc1xDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMsc1[0] = Integer.parseInt(msc1xDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				msc1yDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMsc1[1] = Integer.parseInt(msc1yDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				msc1zDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMsc1[2] = Integer.parseInt(msc1zDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				msc2xDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMsc2[0] = Integer.parseInt(msc2xDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				msc2yDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMsc2[1] = Integer.parseInt(msc2yDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				msc2zDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMsc2[2] = Integer.parseInt(msc2zDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				szxDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defSz[0] = Integer.parseInt(szxDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				szyDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defSz[1] = Integer.parseInt(szyDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				szzDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defSz[2] = Integer.parseInt(szzDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				nmscxDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defNmsc[0] = Integer.parseInt(nmscxDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				nmscyDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defNmsc[1] = Integer.parseInt(nmscyDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				nmsczDefEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defNmsc[2] = Integer.parseInt(nmsczDefEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				defwSigEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defWsigma = Float.parseFloat(defwSigEdit.getText());
						} catch(Exception e) {}
					}
				});
				defwEtaEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defWeta = Float.parseFloat(defwEtaEdit.getText());
						} catch(Exception e) {}
					}
				});
				defnSigEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defNsigma = Float.parseFloat(defnSigEdit.getText());
						} catch(Exception e) {}
					}
				});
				defnEtaEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defNeta = Float.parseFloat(defnEtaEdit.getText());
						} catch(Exception e) {}
					}
				});
				defnMapCtEdit.getDocument().addDocumentListener(new DocumentListener() {
					public void changedUpdate(DocumentEvent e) {update();}
					public void insertUpdate(DocumentEvent e) {update();}
					public void removeUpdate(DocumentEvent e) {update();}
					void update() {
						try {
							defMapCt = Integer.parseInt(defnMapCtEdit.getText());
						} catch(Exception e) {}
					}
				});
				
				break;
		}
		Dimension newsz = paramContainer.getPreferredSize();
		paramEdit.setSize(newsz.width+10, newsz.height+30);
		
		paramEdit.setContentPane(paramContainer);
	}
	
	public void enableSensUpstream(int startNode) {
		DAGnode tnode = nodeArr.get(nodeLookup.get(startNode));
		if(tnode.connOrigin.size()==0) {
			nodeArr.get(nodeLookup.get(startNode)).computeError=true;
		} else {
			nodeArr.get(nodeLookup.get(startNode)).withSens = true;
			for(int i=0; i<tnode.connOrigin.size(); i++) {
				enableSensUpstream(connArr.get(connLookup.get(tnode.connOrigin.get(i))).to);
			}
		}
	}
	
	public void disableSensDownstream(int startNode) {
		DAGnode tnode = nodeArr.get(nodeLookup.get(startNode));
		nodeArr.get(nodeLookup.get(startNode)).withSens = false;
		for(int i=0; i<tnode.connDest.size(); i++) {
			disableSensDownstream(connArr.get(connLookup.get(tnode.connDest.get(i))).from);
		}
	}
	
    public int mapToNode(int x, int y) {
		int mappedTo = -1;
		for(int i=0; i<nodeArr.size(); i++) {
	    	if(nodeArr.get(i).containsPoint(x,y)) {
				mappedTo = i;
				break;
	    	}
		}
		return mappedTo;
    }

    public int mapToConn(int x, int y) {
		int mappedTo = -1;
		for(int i=0; i<connArr.size(); i++) {
	    	if(connArr.get(i).containsPoint(x,y)==1) {
				mappedTo = i;
				break;
	    	}
		}
		return mappedTo;
    }

    public int mapToConnStart(int x, int y) {
    	int mappedTo = -1;
		for(int i=0; i<connArr.size(); i++) {
	    	if(connArr.get(i).from==-1) {
				if(connArr.get(i).containsPoint(x,y)==2) {
					mappedTo = i;
					break;
				}
	    	}
		}
		return mappedTo;
    }
    
    public int mapToConnEnd(int x, int y) {
    	int mappedTo = -1;
		for(int i=0; i<connArr.size(); i++) {
	    	if(connArr.get(i).to==-1) {
	    		if(connArr.get(i).containsPoint(x,y)==3) {
	    			mappedTo = i;
	    			break;
	    		}
	    	}
		}
		return mappedTo;
    }
    
    public void handleMouseMovement(int tx, int ty) {
		switch(mode) {
			case 1: nodeArr.get(nodeLookup.get(selectedItem)).setPos(tx,ty); break;
			case 2: connArr.get(connLookup.get(selectedItem)).setEnd(tx,ty); break;
			case 4: connArr.get(connLookup.get(selectedItem)).setCenter(tx,ty);break;
			case 5: nodeArr.get(nodeLookup.get(selectedItem)).setPos(tx,ty);
	    		for(int i=0; i<nodeArr.get(nodeLookup.get(selectedItem)).connDest.size(); i++) {
                	connArr.get(connLookup.get( nodeArr.get(nodeLookup.get(selectedItem)).connDest.get(i) )).setEnd(tx,ty+6);
            	}
            	for(int i=0; i<nodeArr.get(nodeLookup.get(selectedItem)).connOrigin.size(); i++) {
					connArr.get(connLookup.get( nodeArr.get(nodeLookup.get(selectedItem)).connOrigin.get(i) )).setStart(tx,ty-13);
            	}
	    		break;
			case 6: connArr.get(connLookup.get(selectedItem)).tempMove(tx,ty); break;
			case 7: connArr.get(connLookup.get(selectedItem)).setStart(tx,ty); break;
			case 8: connArr.get(connLookup.get(selectedItem)).setEnd(tx,ty); break;
			default: break;
		}
    }

    public void finalizeElement(int tx, int ty) {
		if(mode == 1) {
	    	nodeArr.get(nodeLookup.get(selectedItem)).setPos(tx,ty);
	    	nodeArr.get(nodeLookup.get(selectedItem)).active = false;
	    	this.redrawBackground(wLast, hLast);
	    	this.modeSet(3);
		} else if(mode == 2) {
			int mappedTo = this.mapToNode(tx,ty);
	    	if(mappedTo != -1) {
				Dimension endCoord = nodeArr.get(mappedTo).getPos();
				
				int from = connArr.get(connLookup.get(selectedItem)).from;
				int to = nodeArr.get(mappedTo).id;
				
				if(!this.checkForLoops(new Vector<Integer>(), to, from)) {
					connArr.get(connLookup.get(selectedItem)).setEnd(endCoord.width, endCoord.height+6);
					connArr.get(connLookup.get(selectedItem)).active = false;
					connArr.get(connLookup.get(selectedItem)).setDest(nodeArr.get(mappedTo).id,nodeArr.get(mappedTo).mapCt);
					nodeArr.get(nodeLookup.get(to)).connDest.add(selectedItem);
					nodeArr.get(nodeLookup.get(from)).connOrigin.add(selectedItem);
					this.modeSet(9);
				} else {
					connArr.remove((int)connLookup.get(selectedItem));
					connLookup.remove(selectedItem);
					this.modeSet(0);
				}
	    	} else {
				connArr.remove((int)connLookup.get(selectedItem));
				connLookup.remove(selectedItem);
				this.modeSet(0);
	    	}
	    	this.redrawBackground(wLast, hLast);
		} else if(mode == 4) {
	    	connArr.get(connLookup.get(selectedItem)).active = false;
	    	this.redrawBackground(wLast, hLast);
	    	this.modeSet(9);
		} else if(mode == 5) {
	    	nodeArr.get(nodeLookup.get(selectedItem)).active = false;
	    	for(int i=0; i<nodeArr.get(nodeLookup.get(selectedItem)).connOrigin.size(); i++) {
				connArr.get(connLookup.get( nodeArr.get(nodeLookup.get(selectedItem)).connOrigin.get(i) )).active = false;
	    	}
	    	for(int i=0; i<nodeArr.get(nodeLookup.get(selectedItem)).connDest.size(); i++) {
				connArr.get(connLookup.get( nodeArr.get(nodeLookup.get(selectedItem)).connDest.get(i) )).active = false;
	    	}
	    	this.redrawBackground(wLast, hLast);
	    	this.modeSet(3);
		} else if(mode == 6) {
			connArr.get(connLookup.get(selectedItem)).active = false;
	    	this.redrawBackground(wLast, hLast);
	    	this.modeSet(9);
		} else if(mode == 7) {
			connArr.get(connLookup.get(selectedItem)).active = false;
			
			int mappedTo = this.mapToNode(tx,ty);
	    	if(mappedTo != -1) {
				Dimension startCoord = nodeArr.get(mappedTo).getPos();
				if(connArr.get(connLookup.get(selectedItem)).to==-1) {
					if(connArr.get(connLookup.get(selectedItem)).mict==nodeArr.get(mappedTo).mapCt) {
						connArr.get(connLookup.get(selectedItem)).setStart(startCoord.width, startCoord.height-13);
						connArr.get(connLookup.get(selectedItem)).setOrigin(nodeArr.get(mappedTo).id);
						nodeArr.get(mappedTo).connOrigin.add(selectedItem);
					}
				} else {
					if(!this.checkForLoops(new Vector<Integer>(), connArr.get(connLookup.get(selectedItem)).to, nodeArr.get(mappedTo).id)) {
						if(connArr.get(connLookup.get(selectedItem)).mict==nodeArr.get(mappedTo).mapCt) {
							connArr.get(connLookup.get(selectedItem)).setStart(startCoord.width, startCoord.height-13);
							connArr.get(connLookup.get(selectedItem)).setOrigin(nodeArr.get(mappedTo).id);
							nodeArr.get(mappedTo).connOrigin.add(selectedItem);
						}
					}
				}
	    	}
	    	
	    	this.redrawBackground(wLast, hLast);
	    	this.modeSet(9);
		} else if(mode == 8) {
			connArr.get(connLookup.get(selectedItem)).active = false;
			
			int mappedTo = this.mapToNode(tx,ty);
	    	if(mappedTo != -1) {
				Dimension endCoord = nodeArr.get(mappedTo).getPos();
				if(connArr.get(connLookup.get(selectedItem)).from==-1) {
					if(connArr.get(connLookup.get(selectedItem)).moct==nodeArr.get(mappedTo).mapCt) {
						connArr.get(connLookup.get(selectedItem)).setEnd(endCoord.width, endCoord.height+6);
						connArr.get(connLookup.get(selectedItem)).setDest(nodeArr.get(mappedTo).id,nodeArr.get(mappedTo).mapCt);
						nodeArr.get(mappedTo).connDest.add(selectedItem);
					}
				} else {
					if(!this.checkForLoops(new Vector<Integer>(), nodeArr.get(mappedTo).id, connArr.get(connLookup.get(selectedItem)).from)) {
						if(connArr.get(connLookup.get(selectedItem)).moct==nodeArr.get(mappedTo).mapCt) {
							connArr.get(connLookup.get(selectedItem)).setEnd(endCoord.width, endCoord.height+6);
							connArr.get(connLookup.get(selectedItem)).setDest(nodeArr.get(mappedTo).id,nodeArr.get(mappedTo).mapCt);
							nodeArr.get(mappedTo).connDest.add(selectedItem);
						}
					}
				}
	    	}
			
			this.redrawBackground(wLast, hLast);
			this.modeSet(9);
		}
    }
    
    public void addNode(int x, int y, Component gContext, int overrideId) {
		this.unselectAll();
		if(overrideId != -1) {
			selectedItem = overrideId;
		} else {
			selectedItem = nextNodeId;
		}
		nodeArr.add(new DAGnode(selectedItem,x,y,gContext,defMapCt,defNsigma,defNeta,defNmsc));// add a node
		nodeLookup.put(selectedItem, nodeArr.size()-1);
		nextNodeId = Math.max(nextNodeId,overrideId)+1;
		this.modeSet(1); // addnode mode
    }
    
    public void addNode(DAGnode tnode) {
    	this.unselectAll();
    	selectedItem = nextNodeId;
    	tnode.id = selectedItem;
    	nodeArr.add(tnode);
		nodeLookup.put(selectedItem, nodeArr.size()-1);
		nextNodeId++;
		this.modeSet(1);
    }
    
    public void addConn(DAGconn tconn) {
    	this.unselectAll();
    	selectedItem = nextConnId;
    	tconn.id = selectedItem;
    	connArr.add(tconn);
    	connLookup.put(selectedItem,connArr.size()-1);
    	nextConnId++;
    	this.modeSet(6); // drag center of node mode?
    }
    
    public void addConn(int startNode, int overrideId) {
		this.unselectAll();
		if(!nodeArr.get(startNode).computeError) {
			if(overrideId != -1) {
				selectedItem = overrideId;
			} else {
				selectedItem = nextConnId;
			}
			Dimension startCoord = nodeArr.get(startNode).getPos();
			int mict = nodeArr.get(startNode).mapCt;
			connArr.add(new DAGconn(selectedItem, nodeArr.get(startNode).id, startCoord.width, startCoord.height-13, mict, defSz, defMsc1, defMsc2, defWsigma, defWeta));// add a conn
			connLookup.put(selectedItem,connArr.size()-1);
			nextConnId = Math.max(nextConnId, overrideId)+1;
			this.modeSet(2); // addconn mode
    	}
    }
    
    public void addConn(int startNodeId, int destNodeId, int overrideId) {
		this.unselectAll();
		if(overrideId != -1) {
			selectedItem = overrideId;
		} else {
			selectedItem = nextConnId;
		}
		Dimension startCoord = nodeArr.get((int)nodeLookup.get(startNodeId)).getPos();
		Dimension endCoord = nodeArr.get((int)nodeLookup.get(destNodeId)).getPos();
		int mict = nodeArr.get((int)nodeLookup.get(startNodeId)).mapCt;
		int moct = nodeArr.get((int)nodeLookup.get(destNodeId)).mapCt;
		connArr.add(new DAGconn(selectedItem, startNodeId, startCoord.width, startCoord.height-13,destNodeId,endCoord.width,endCoord.height+6, mict, moct, defSz, defMsc1, defMsc2, defWsigma, defWeta));
		connLookup.put(selectedItem,connArr.size()-1);
		nextConnId = Math.max(nextConnId, overrideId)+1;
		nodeArr.get((int)nodeLookup.get(destNodeId)).connDest.add(selectedItem);
		nodeArr.get((int)nodeLookup.get(startNodeId)).connOrigin.add(selectedItem);
		this.modeSet(2); // addconn mode
    }

    public void deleteSelected() {
		for(int i=0; i<nodeArr.size(); i++) {
	    	if(nodeArr.get(i).selected == true) {
	    		disableSensDownstream(nodeArr.get(i).id);
				removeNode(nodeArr.get(i).id);
	    	}
		}
		for(int i=0; i<connArr.size(); i++) {
	    	if(connArr.get(i).selected == true) {
	    		if(connArr.get(i).to != -1) {
	    			disableSensDownstream(connArr.get(i).from);
	    		}
	    		removeConn(connArr.get(i).id);
	    	}
		}
		this.modeSet(0);
		this.redrawBackground(wLast, hLast);
	}

    public void unselectAll() {
		for(int i=0; i<nodeArr.size(); i++) {
	    	nodeArr.get(i).selected = false;
	    	nodeArr.get(i).active = false;
		}
		for(int i=0; i<connArr.size(); i++) {
	    	connArr.get(i).selected = false;
	    	connArr.get(i).active = false;
		}
		this.redrawBackground(wLast, hLast);
		this.modeSet(0);
    }
    
    public void pushNodeToTop(int chooseNode) {
    	for(int i=chooseNode; i<nodeArr.size()-1; i++) {
			nodeLookup.put(nodeArr.get(i+1).id,i);
			Collections.swap(nodeArr,i+1,i);
		}
		nodeLookup.put(nodeArr.get(nodeArr.size()-1).id, nodeArr.size()-1);
    }
    
    public void pushConnToTop(int chooseConn) {
    	for(int i=chooseConn; i<connArr.size()-1; i++) {
			connLookup.put(connArr.get(i+1).id,i);
			Collections.swap(connArr,i+1,i);
		}
		connLookup.put(connArr.get(connArr.size()-1).id, connArr.size()-1);
    }
    
    public void selectNode(int chooseNode) {
		this.unselectAll();
		pushNodeToTop(chooseNode);
		
		nodeArr.get(nodeArr.size()-1).active = true;
		nodeArr.get(nodeArr.size()-1).selected = true;
		for(int i=0; i<nodeArr.get(nodeArr.size()-1).connOrigin.size(); i++) {
			int cid = nodeArr.get(nodeArr.size()-1).connOrigin.get(i);
	    	connArr.get(connLookup.get(cid)).active = true;
	    	pushConnToTop(cid);
		}
		for(int i=0; i<nodeArr.get(nodeArr.size()-1).connDest.size(); i++) {
			int cid = connLookup.get( nodeArr.get(nodeArr.size()-1).connDest.get(i) );
	    	connArr.get(cid).active = true;
	    	pushConnToTop(cid);
		}
		
		this.redrawBackground(wLast, hLast);
		selectedItem = nodeArr.get(nodeArr.size()-1).id;
		this.modeSet(5);
    }

    public void selectConn(int chooseConn) {
		this.unselectAll();
		
		pushConnToTop(chooseConn);
		
		connArr.get(connArr.size()-1).active = true;
		connArr.get(connArr.size()-1).selected = true;
		this.redrawBackground(wLast, hLast);
		selectedItem = connArr.get(connArr.size()-1).id;
		this.modeSet(4);
    }
    
    public void selectConnEnd(int chooseConn) {
    	this.unselectAll();
    	
    	pushConnToTop(chooseConn);
    	
		connArr.get(connArr.size()-1).active = true;
		connArr.get(connArr.size()-1).selected = true;
		this.redrawBackground(wLast, hLast);
		selectedItem = connArr.get(connArr.size()-1).id;
		this.modeSet(8);
    }
    
    public void selectConnStart(int chooseConn) {
    	this.unselectAll();
    	
    	pushConnToTop(chooseConn);
    	
		connArr.get(connArr.size()-1).active = true;
		connArr.get(connArr.size()-1).selected = true;
		this.redrawBackground(wLast, hLast);
		selectedItem = connArr.get(connArr.size()-1).id;
		this.modeSet(7);
    }
    
    public void removeNode(int id) {
		DAGnode tNode = nodeArr.get((int)nodeLookup.get(id));
		while(tNode.connOrigin.size()>0) {
			this.removeConn(tNode.connOrigin.get(0));
		}
		while(tNode.connDest.size()>0) {
			this.removeConn(tNode.connDest.get(0));
		}
		int removedNode = nodeLookup.get(id);
		nodeArr.remove((int)removedNode);
		nodeLookup.remove((Integer)id);
		Enumeration<Integer> nodeKeys = nodeLookup.keys();
		int key;
		while(nodeKeys.hasMoreElements()) {
	    	key = (Integer)nodeKeys.nextElement();
	    	if(nodeLookup.get(key)>removedNode) {
				nodeLookup.put(key,nodeLookup.get(key)-1); // left shift 1 (removed one node)
	    	}
		}
    }

    public void removeConn(int id) {
		int removedConn = (int)connLookup.get(id);
		int from = connArr.get(removedConn).from;
		int to = connArr.get(removedConn).to;
		connArr.remove(removedConn);
		connLookup.remove((Integer)id);
		Enumeration<Integer> connKeys = connLookup.keys();
		int key;
		while(connKeys.hasMoreElements()) {
	    	key = (Integer)connKeys.nextElement();
	    	if(connLookup.get(key)>removedConn) {
				connLookup.put(key,connLookup.get(key)-1);
	    	}
		}
		this.redrawBackground(wLast, hLast);
		
		if(to != -1) {
			nodeArr.get(nodeLookup.get(to)).connDest.remove((Integer)id);
		}
		if(from != -1) {
			nodeArr.get(nodeLookup.get(from)).connOrigin.remove((Integer)id);
		}
    }
    
    public void changeMapCt(int nodeId, int newCt, float sigma) {
    	nodeArr.get((int)nodeLookup.get(nodeId)).setMapCt(newCt);
    	DAGnode tnode = nodeArr.get((int)nodeLookup.get(nodeId));
    	for(int i=0; i<tnode.connOrigin.size(); i++) {
    		connArr.get((int)connLookup.get(tnode.connOrigin.get(i))).changeSize(newCt,-1,-1,-1,-1);
    	}
    	for(int i=0; i<tnode.connDest.size(); i++) {
    		connArr.get((int)connLookup.get(tnode.connDest.get(i))).changeSize(-1,-1,-1,-1,newCt);
    	}
    }
    
    public void changeBias(int nodeId, float bias[]) {
    	nodeArr.get(nodeLookup.get(nodeId)).setBias(bias);
    }
    
    public void changeWeight(int connId, int wsx, int wsy, int wsz, float tdata[]) {
    	int[] sz = {wsx,wsy,wsz};
    	connArr.get(connLookup.get(connId)).setVals(sz, tdata);
    }
    
    public void renameNode(int nodeId, String newName) {
    	nodeArr.get(nodeLookup.get(nodeId)).renameNode(newName);
    }
    
    public void renameConn(int connId, String newName) {
    	connArr.get(connLookup.get(connId)).name = newName;
    }
    
    public void changeNodeMSC(int nodeId, int xs, int ys, int zs) {
    	nodeArr.get(nodeLookup.get(nodeId)).msc[0]=xs;
    	nodeArr.get(nodeLookup.get(nodeId)).msc[1]=ys;
    	nodeArr.get(nodeLookup.get(nodeId)).msc[2]=zs;
    }
    
    public void changeConnMSC1(int connId, int xs, int ys, int zs) {
    	connArr.get(connLookup.get(connId)).msc1[0]=xs;
    	connArr.get(connLookup.get(connId)).msc1[1]=ys;
    	connArr.get(connLookup.get(connId)).msc1[2]=zs;
    }
    
    public void changeConnMSC2(int connId, int xb, int yb, int zb) {
    	connArr.get(connLookup.get(connId)).msc2[0]=xb;
    	connArr.get(connLookup.get(connId)).msc2[1]=yb;
    	connArr.get(connLookup.get(connId)).msc2[2]=zb;
    }
    
    public void setConnVals(int connId, int xsz, int ysz, int zsz, float newVals[]) {
    	int sz[] = {xsz,ysz,zsz};
    	connArr.get(connLookup.get(connId)).setVals(sz,newVals);
    }
    
    public void changeNodeEta(int nodeId, float eta) {
    	nodeArr.get(nodeLookup.get(nodeId)).eta = eta;
    }
    
    public void changeConnEta(int connId, float eta) {
    	connArr.get(connLookup.get(connId)).eta = eta;
    }
    
    public void redrawBackground(int w, int h) {
		wLast = w;
		hLast = h;
        staticElements = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2a = staticElements.createGraphics();
        g2a.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                             RenderingHints.VALUE_ANTIALIAS_ON);

        g2a.setPaint(Color.white);
        g2a.fillRect(0,0,w,h);

		Graphics g = (Graphics)g2a;
		for(int i=0; i<connArr.size(); i++) {
	    	if(connArr.get(i).active == false) {
				connArr.get(i).selfDraw(g);
	    	}
        }
		for(int i=0; i<nodeArr.size(); i++) {
		    if(nodeArr.get(i).active == false) {
				nodeArr.get(i).selfDraw(g);
		    }
	    }
        g2a.dispose();
    }

    public void redraw(Graphics g) {
		g.drawImage(staticElements,0,0,null);
		for(int i=0; i<connArr.size(); i++) {
	    	if(connArr.get(i).active == true) {
				connArr.get(i).selfDraw(g);
	    	}
		}
		for(int i=0; i<nodeArr.size(); i++) {
	    	if(nodeArr.get(i).active == true) {
				nodeArr.get(i).selfDraw(g);
	    	}
		}
		
    }
    
    public void drawFresh(Graphics g) {
    	Graphics2D g2 = (Graphics2D)g;
    	g2.setPaint(Color.white);
        g2.fillRect(0,0,wLast,hLast);
        for(int i=0; i<connArr.size(); i++) {
	    	connArr.get(i).selfDraw(g);
	    }
        for(int i=0; i<nodeArr.size(); i++) {
			nodeArr.get(i).selfDraw(g);
		}
    }
}