import java.awt.Color;
import java.awt.Cursor;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;

import javax.swing.BorderFactory;
import javax.swing.JPanel;
import javax.swing.JToggleButton;

public class DAGPanel extends JPanel implements Printable {
	public DAGgraph dagGraph;
	private static final long serialVersionUID = 1L;
	Cursor curCursor;
	JToggleButton buttonInfo, buttonAddConn, buttonAddActivity;
	
	public DAGPanel(JToggleButton tButtonInfo, JToggleButton tButtonAddConn, JToggleButton tButtonAddActivity) {
		buttonInfo = tButtonInfo;
		buttonAddConn = tButtonAddConn;
		buttonAddActivity = tButtonAddActivity;
    	setBackground(Color.white);
    	setBorder(BorderFactory.createLineBorder(Color.black));
    	addMouseListener(new DAGMouseListener());
    	addMouseMotionListener(new DAGMouseMotionListener());
    	try {
    		dagGraph = new DAGgraph(buttonInfo, this);
    	} catch(Exception e) {
    		e.printStackTrace();
    	}
    	
	}

	public void paint(Graphics g) {
    	super.paintComponent(g);
    	
    	dagGraph.redraw(g);
    
    	if (curCursor != null) {
			setCursor(curCursor);
    	}
	}
	
	public int print(Graphics g, PageFormat pf, int page) throws PrinterException {
		if (page > 0) {
			return Printable.NO_SUCH_PAGE;
		}
		
		Graphics2D g2d = (Graphics2D)g;
		g2d.translate(pf.getImageableX(), pf.getImageableY());
		
		/* Print the entire visible contents of a java.awt.Frame */
		dagGraph.drawFresh(g);
		return Printable.PAGE_EXISTS;
	}

	class DAGMouseListener extends MouseAdapter {
    	public void mousePressed(MouseEvent e) {
			if(buttonAddConn.isSelected()) {
	    		int node = dagGraph.mapToNode(e.getX(), e.getY());
	    		if(node != -1) {
					dagGraph.addConn(node,-1);
	    		}
			} else if(buttonAddActivity.isSelected()) {
	    		dagGraph.addNode(e.getX(), e.getY(), buttonInfo,-1);
			} else {
				int mappedTo = dagGraph.mapToConnStart(e.getX(), e.getY());
				if(mappedTo == -1) {
					mappedTo = dagGraph.mapToConnEnd(e.getX(), e.getY());
					if(mappedTo == -1) {
						mappedTo = dagGraph.mapToConn(e.getX(), e.getY());
						if(mappedTo == -1) {
							mappedTo = dagGraph.mapToNode(e.getX(), e.getY());
							if(mappedTo == -1) {
								dagGraph.unselectAll();// click mapped to nothing
							} else {
								dagGraph.selectNode(mappedTo);
							}
						} else {
							dagGraph.selectConn(mappedTo);
						}
					} else {
						dagGraph.selectConnEnd(mappedTo);
					}
				} else {
					dagGraph.selectConnStart(mappedTo);
				}
			}
			repaint();
    	}
    
		public void mouseReleased(MouseEvent e) {
			dagGraph.finalizeElement(e.getX(), e.getY());
			repaint();
		}
		public void mouseClicked(MouseEvent e) { }
	}

	class DAGMouseMotionListener extends MouseMotionAdapter {
    	public void mouseDragged(MouseEvent e) {
			dagGraph.handleMouseMovement(e.getX(), e.getY());
			repaint();
    	}
    	public void mouseMoved(MouseEvent e) {
			dagGraph.handleMouseMovement(e.getX(), e.getY());
			repaint();
    	}
	}
}