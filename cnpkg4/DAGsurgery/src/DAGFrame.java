import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;

import javax.swing.Box;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JToggleButton;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;

class DAGFrame extends JFrame {
	private static final long serialVersionUID = 1L;
	
	public DAGPanel canvas;
    JToggleButton buttonAddActivity, buttonAddConn;

	JToggleButton buttonInfo;

	JButton buttonOrganize;
    
    public DAGFrame(String dagname) {
		super();
		setTitle("DAGnabbit - "+dagname);
		Container container = getContentPane();
		
		JToolBar dagbar = new JToolBar("Tools",JToolBar.HORIZONTAL);
		buttonAddActivity = new JToggleButton("Add activity");
		dagbar.add(buttonAddActivity);
		
		buttonAddConn = new JToggleButton("Add weight");
		dagbar.add(buttonAddConn);
		
		buttonInfo = new JToggleButton("Parameters");
		dagbar.add(Box.createHorizontalGlue());
		dagbar.add(buttonInfo);
		
		canvas = new DAGPanel(buttonInfo, buttonAddConn, buttonAddActivity);
		
		JPanel panel = new JPanel();
		panel.setLayout(new GridLayout(1,2));
		
		container.add(dagbar,BorderLayout.NORTH);
		container.add(canvas);
		container.add(panel,BorderLayout.SOUTH);
	
		setFocusable(true);
		addKeyListener(new DAGKeyListener());
		panel.addKeyListener(new DAGKeyListener());
		dagbar.addKeyListener(new DAGKeyListener());
		buttonAddActivity.addKeyListener(new DAGKeyListener());
		buttonAddConn.addKeyListener(new DAGKeyListener());
		buttonInfo.addKeyListener(new DAGKeyListener());
		panel.addComponentListener(new DAGComponentListener());
		buttonAddConn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if(buttonAddConn.isSelected()) {
					buttonAddActivity.setSelected(false);
				}
			}
		});
		buttonAddActivity.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if(buttonAddActivity.isSelected()) {
					buttonAddConn.setSelected(false);
				}
			}
		});
		
		buttonInfo.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					if(buttonInfo.isSelected()) {
						canvas.dagGraph.showParamEdit(true);
					} else {
						canvas.dagGraph.showParamEdit(false);
					}
				}
			});
		
		setSize(500,500);
		System.setProperty("apple.laf.useScreenMenuBar", "true");
		
		JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        JMenu fileMenu = new JMenu("File");
        JMenu editMenu = new JMenu("Edit");
        menuBar.add(fileMenu);
        menuBar.add(editMenu);
        
        Toolkit tk = Toolkit.getDefaultToolkit();
        
        JMenuItem closeAction = new JMenuItem("Close");
        closeAction.setAccelerator(KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_W, tk.getMenuShortcutKeyMask()));
        JMenuItem printAction = new JMenuItem("Print");
        printAction.setAccelerator(KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_P, tk.getMenuShortcutKeyMask()));
        JMenuItem cutAction = new JMenuItem("Cut");
        cutAction.setAccelerator(KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_X, tk.getMenuShortcutKeyMask()));
        JMenuItem copyAction = new JMenuItem("Copy");
        copyAction.setAccelerator(KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_C, tk.getMenuShortcutKeyMask()));
        JMenuItem pasteAction = new JMenuItem("Paste");
        pasteAction.setAccelerator(KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_V, tk.getMenuShortcutKeyMask()));
        JMenuItem reorganizeAction = new JMenuItem("Auto Layout");
        reorganizeAction.setAccelerator(KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_R, tk.getMenuShortcutKeyMask()+java.awt.event.InputEvent.SHIFT_MASK));
        
        fileMenu.add(printAction);
        fileMenu.add(closeAction);
        editMenu.add(cutAction);
        editMenu.add(copyAction);
        editMenu.add(pasteAction);
        editMenu.add(new JSeparator());
        editMenu.add(reorganizeAction);
        
        printAction.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				 PrinterJob pj = PrinterJob.getPrinterJob();
				 pj.setPrintable(canvas);
				     if (pj.printDialog()) {
				         try {pj.print();}
					 catch (PrinterException exc) {
					   System.out.println(exc);
					  }
				}
            }
        });
        
        closeAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                dispose();
            }
        });
        
        copyAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.dagGraph.copySelected();
            }
        });
        
        cutAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.dagGraph.cutSelected();
            }
        });
        
        pasteAction.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.dagGraph.pasteSelected();
            }
        });
        
        reorganizeAction.addActionListener(new ActionListener() {
        	public void actionPerformed(ActionEvent e) {
        		canvas.dagGraph.selfOrganize(true);
        	}
        });
        
        setVisible(true);
    }
    
    public void cleanup() {
    	canvas.dagGraph.selfOrganize(false);
    }
    
    public void makeClean() {
    	canvas.dagGraph.selfOrganize(true);
    }
    
    public void addSens(int mappedToId) {
    	canvas.dagGraph.addSensToNode(mappedToId);
    }
    
    public void addErr(int mappedToId) {
    	canvas.dagGraph.addErrToNode(mappedToId);
    }
    
    public void addWeight(int from, int to, String layerName, int tid, int dimx, int dimy, int dimz, float weight[], int xs, int ys, int zs, int xb, int yb, int zb, float eta) {
    	canvas.dagGraph.addConn(from, to, tid);
    	canvas.dagGraph.renameConn(tid, layerName);
    	canvas.dagGraph.changeConnMSC1(tid, xs, ys, zs);
    	canvas.dagGraph.changeConnMSC2(tid, xb, yb, zb);
    	canvas.dagGraph.changeConnEta(tid, eta);
    	// Havent set values or size yet
    	canvas.dagGraph.setConnVals(tid,dimx,dimy,dimz,weight);
    	canvas.dagGraph.unselectAll();
    }
    
    public void addComputed(String layerName, int tid, int fmCt, float bias[], int ys, int xs, int zs, float eta) {
    	canvas.dagGraph.addNode(100, 100, canvas, tid);
    	canvas.dagGraph.renameNode(tid, layerName);
    	canvas.dagGraph.changeMapCt(tid, fmCt, 0);
    	canvas.dagGraph.changeBias(tid, bias);
    	canvas.dagGraph.changeNodeMSC(tid, xs, ys, zs);
    	canvas.dagGraph.changeNodeEta(tid, eta);
    	canvas.dagGraph.unselectAll();
    }
    
    public void addUncomputed(String layerName, int tid, int fmCt, int ys, int xs, int zs) {
    	canvas.dagGraph.addNode(100, 100, canvas, tid);
    	canvas.dagGraph.renameNode(tid, layerName);
    	canvas.dagGraph.changeMapCt(tid, fmCt, 0);
    	canvas.dagGraph.changeNodeMSC(tid, xs, ys, zs);
    	canvas.dagGraph.unselectAll();
    }
    
    public int numUncomputed() {
    	return canvas.dagGraph.numUncomputed();
    }
    
    public int numComputed() {
    	return canvas.dagGraph.numComputed();
    }
    
    public int numConn() {
    	return canvas.dagGraph.numConn();
    }
    
    public char[] getUncomputedName(int nodeId) {
    	return canvas.dagGraph.getUncomputedName(nodeId);
    }
    
    public char[] getComputedName(int nodeId) {
    	return canvas.dagGraph.getComputedName(nodeId);
    }
    
    public char[] getConnName(int connId) {
    	return canvas.dagGraph.getConnName(connId);
    }
    
    public int getUncomputedMapCt(int nodeId) {
    	return canvas.dagGraph.getUncomputedMapCt(nodeId);
    }
    
    public int getComputedMapCt(int nodeId) {
    	return canvas.dagGraph.getComputedMapCt(nodeId);
    }

    public int getUncomputedId(int nodeId) {
    	return canvas.dagGraph.getUncomputedId(nodeId);
    }
    
    public int getComputedId(int nodeId) {
    	return canvas.dagGraph.getComputedId(nodeId);
    }

    public int[] getUncomputedSpace(int nodeId) {
    	return canvas.dagGraph.getSpace(nodeId,false);
    }
    
    public int[] getComputedSpace(int nodeId) {
    	return canvas.dagGraph.getSpace(nodeId,true);
    }
    
    public int[] getConnSpace(int connId) {
    	return canvas.dagGraph.getConnSpace(connId);
    }
    
    public int[] getConnBlock(int connId) {
    	return canvas.dagGraph.getConnBlock(connId);
    }
    
    public int[] getConnSize(int connId) {
    	return canvas.dagGraph.getConnSize(connId);
    }
    
    public float[] getConnVals(int connId) {
    	return canvas.dagGraph.getConnVals(connId);
    }
    
    public int getConnFrom(int connId) {
    	return canvas.dagGraph.getConnFrom(connId);
    }
    
    public int getConnTo(int connId) {
    	return canvas.dagGraph.getConnTo(connId);
    }
    
    public float getConnEta(int connId) {
    	return canvas.dagGraph.getConnEta(connId);
    }
    
    public float[] getComputedBias(int nodeId) {
    	return canvas.dagGraph.getComputedBias(nodeId);
    }
    
    public float getComputedEta(int nodeId) {
    	return canvas.dagGraph.getComputedEta(nodeId);
    }

    public boolean computedHasSens(int nodeId) {
    	return canvas.dagGraph.computedHasSens(nodeId);
    }
    
    public boolean computedHasErr(int nodeId) {
    	return canvas.dagGraph.computedHasErr(nodeId);
    }
    
    class DAGComponentListener implements ComponentListener {
		public void componentResized(ComponentEvent e) {
	    	Dimension rv  = e.getComponent().getParent().getSize();
	    	canvas.dagGraph.redrawBackground(rv.width,rv.height);
	    	canvas.repaint();
		}
		public void componentHidden(ComponentEvent e) {}
		public void componentMoved(ComponentEvent e) {}
		public void componentShown(ComponentEvent e) {}
    }
    
    class DAGKeyListener implements KeyListener {
		public void keyPressed(KeyEvent e) {
	    	if (e.getKeyCode() == KeyEvent.VK_CONTROL) {
				buttonAddActivity.setSelected(true);
				buttonAddConn.setSelected(false);
	    	} else if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
				buttonAddConn.setSelected(true);
				buttonAddActivity.setSelected(false);
	    	}
		}
		public void keyReleased(KeyEvent e) {
	    	if (e.getKeyCode() == KeyEvent.VK_CONTROL) {
				buttonAddActivity.setSelected(false);
	    	} else if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
				buttonAddConn.setSelected(false);
	    	} else if (e.getKeyCode() == KeyEvent.VK_DELETE) {
	    		canvas.dagGraph.deleteSelected();
				canvas.repaint();
	    	}
		}
		public void keyTyped(KeyEvent e) {}
	}
}