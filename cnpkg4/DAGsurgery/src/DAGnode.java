import java.util.Random;
import java.util.Vector;
import java.awt.geom.RoundRectangle2D;
import java.awt.*;

public class DAGnode extends DAGelement {
    public Vector<Integer>connOrigin;
    public Vector<Integer>connDest;
    public Boolean withSens, computeError;
	public int mapCt;
	public int width;
	FontMetrics fm;
	public int msc[];
	public float[] bias;
	
	public DAGnode(int tx, int ty, int twidth) {
		super("",0,0,-1,false,false);
		xc=tx;
		yc=ty;
		width=twidth;
		msc = new int[3];
	}
	
    public DAGnode(int tid, int sx, int sy, Component gContext, int iMapCt, float iSigma, float iEta, int iSpace[]) {
    	super("Activity "+tid, iSigma, iEta, tid, true, true);
    	Random generator = new Random();
    	
    	msc = new int[3];
    	
		computeError = false;
		withSens = false;
		
		withSens = false;
		
		connOrigin = new Vector<Integer>();
		connDest = new Vector<Integer>();
		
		
		Font f = new Font("Verdana", Font.PLAIN, 12);
		fm = gContext.getFontMetrics(f);
		width = fm.stringWidth(name)+5;
		
		xc = sx - width/2 -7;
		yc = sy - 9;
		
		mapCt = iMapCt;
		bias = new float[mapCt];
    	for(int i=0; i<mapCt; i++) {
    		bias[i] = sigma*((float)(generator.nextGaussian()));
    	}
    	for(int i=0; i<3; i++)
    		msc[i]=iSpace[i];
    }
    
    public DAGnode clone() {
    	DAGnode tnode = new DAGnode(xc,yc,width);
    	tnode.computeError = computeError;
    	tnode.withSens = withSens;
    	tnode.fm = fm;
    	tnode.mapCt = mapCt;
    	tnode.bias = new float[mapCt];
    	System.arraycopy(bias, 0, tnode.bias, 0, mapCt);
    	tnode.name = name;
    	tnode.connOrigin = new Vector<Integer>();
    	tnode.connDest = new Vector<Integer>();
    	tnode.eta = eta;
    	for(int i=0; i<3; i++)
    		tnode.msc[i] = this.msc[i];
    	
    	return tnode;
    }
    
    public void renameNode(String tname) {
		int tx = xc + width/2+7;
		name = tname;
		width = fm.stringWidth(name)+5;
		xc = tx - width/2-7;
    }
    
    public void setMapCt(int newMapCt){
    	Random generator = new Random();
    	float tbias[] = new float[newMapCt];
    	for(int i=0; i<newMapCt; i++) {
    		if(i<mapCt) {
    			tbias[i] = bias[i];
    		} else {
    			tbias[i] = sigma*((float)(generator.nextGaussian()));
    		}
    	}
    	mapCt = newMapCt;
    	bias = tbias;
    }
    
    public void reInitialize() {
    	Random generator = new Random();
    	for(int i=0; i<mapCt; i++) {
    		bias[i] = sigma*((float)(generator.nextGaussian()));
    	}
    }
    
    public void setBias(float tbias[]) {
    	for(int i=0; i<mapCt; i++) {
    		bias[i] = tbias[i];
    	}
    }
    
    public void setPos(int tx, int ty) {
		xc = tx - width/2 -7;
		yc = ty - 9;
    }
    
    public Dimension getPos() {
		return(new Dimension(xc+width/2+7,yc+9));
    }
    
    public boolean containsPoint(int tx, int ty) {
		RoundRectangle2D tempNode = new RoundRectangle2D.Float(xc,yc,width+14,15,15,15);
		boolean clicked=false;
		if(tempNode.contains((double)tx,(double)ty)) {
	    	clicked=true;
		}
		return clicked;
    }

    public void selfDraw(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
			    			RenderingHints.VALUE_ANTIALIAS_ON);
		
		if(this.computeError) {
			RoundRectangle2D tmpLabel = new RoundRectangle2D.Float(xc,yc-25,width+14,15+24,15,15);
			g2.setColor(new Color(200,255,200));
			g2.fill(tmpLabel);
			g2.setColor(Color.black);
			g2.draw(tmpLabel);
			g.setFont(new Font("Verdana", Font.PLAIN, 10));
			g.drawString("label",width/2-3+xc,yc+12-26);
		
			RoundRectangle2D tmpErr = new RoundRectangle2D.Float(xc,yc-13,width+14,15+12,15,15);
			g2.setColor(new Color(255,255,200));
			g2.fill(tmpErr);
			g2.setColor(Color.black);
			g2.draw(tmpErr);
			g.setFont(new Font("Verdana", Font.PLAIN, 10));
			g.drawString("error",width/2-3+xc,yc+12-14);
		} else {
			int[] xs = { xc+width/2+7-4, xc+width/2+7, xc+width/2+7+4 };
			int[] ys = { yc, yc-4, yc };
			Polygon triangle = new Polygon(xs, ys, xs.length);
			g2.setColor(Color.black);
			g.fillPolygon(triangle);
		}
		g.setFont(new Font("Verdana", Font.PLAIN, 12));
		if(this.withSens) {
			RoundRectangle2D tmpSens = new RoundRectangle2D.Float(xc-4,yc+4,width+14,15,15,15);
			g2.setColor(new Color(200,200,255));
			g2.fill(tmpSens);
			g2.setColor(Color.black);
			g2.draw(tmpSens);
    	}
		
		RoundRectangle2D tempActivity = new RoundRectangle2D.Float(xc,yc,width+14,15,15,15);
		g2.setColor(Color.lightGray);
		g2.fill(tempActivity);
		if(!selected) {
	    	g2.setColor(Color.black);
		} else {
	    	g2.setColor(Color.yellow);
		}
		g2.draw(tempActivity);
		g2.setColor(Color.black);
		g.setFont(new Font("Verdana", Font.PLAIN, 12));
		g.drawString(name,xc+7,yc+12);
		g.setFont(new Font("Verdana", Font.PLAIN, 9));
		g.drawString(Integer.toString(mapCt),xc+width+18,yc+11);
    }
}