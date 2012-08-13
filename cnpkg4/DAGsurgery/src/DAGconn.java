import java.lang.Math;
import java.util.Random;
import java.awt.geom.CubicCurve2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.awt.*;

public class DAGconn extends DAGelement {
    private int x1,y1,x2,y2,cx1,cy1,cx2,cy2;
    public int from, to;
    
	public Integer mict;
	public Integer moct;
    public int sz[];
    public int[] msc1;
    public int msc2[];
	public float[] data;
	public float centerPushx, centerPushy;
	
    public DAGconn(int id, int sn, int sx, int sy, int mict, int iSz[], int iMsc1[], int iMsc2[], float iSigma, float iEta) {
    	super("weight", iSigma, iEta, id, true, true);
    	
    	centerPushx = centerPushy = 0;
    	
		this.mict = mict;
        
		x2 = x1 = sx;
		y2 = y1 = sy;

		sz = new int[3];
		msc1 = new int[3];
		msc2 = new int[3];
		for(int i=0;i<3;i++) {
			sz[i] = iSz[i];
			msc1[i] = iMsc1[i];
			msc2[i] = iMsc2[i];
		}
		this.selfAdjust();
		from = sn; to = -1;
    }
    
    public DAGconn(int id, int sn, int sx, int sy, int dn, int dx, int dy, int mict, int moct, int sz[], int msc1[], int msc2[], float sigma, float eta) {
		super("weight", sigma, eta, id, false, false);
		
		centerPushx = centerPushy = 0;
		
		this.mict = mict; this.moct = moct;
        x1 = sx; x2 = dx;
		y1 = sy; y2 = dy;
		this.sz = new int[3];
		this.msc1 = new int[3];
		this.msc2 = new int[3];
		for(int i=0;i<3;i++) {
			this.sz[i] = sz[i];
			this.msc1[i] = msc1[i];
			this.msc2[i] = msc2[i];
		}
		this.selfAdjust();
		from = sn; to = dn;
		data = new float[mict*moct*sz[0]*sz[1]*sz[2]];
		this.reInitialize();
    }
    
    public DAGconn() {
    	super("", 0, 0, -1, true, true);
    	centerPushx = centerPushy = 0;
        from = -1;
        to = -1;
		sz = new int[3];
		msc1 = new int[3];
		msc2 = new int[3];
    }
    
    public DAGconn clone() {
    	DAGconn tconn = new DAGconn();
    	tconn.name = name;
    	System.arraycopy(sz, 0, tconn.sz, 0, 3);
    	System.arraycopy(msc1, 0, tconn.msc1, 0, 3);
    	System.arraycopy(msc2, 0, tconn.msc2, 0, 3);
    	tconn.eta = eta;
    	tconn.mict = mict;
    	tconn.moct = moct;
    	tconn.sigma = sigma;
    	tconn.data = new float[mict*moct*sz[0]*sz[1]*sz[2]];
    	System.arraycopy(data, 0, tconn.data, 0, mict*moct*sz[0]*sz[1]*sz[2]);
    	return tconn;
    }
    
    public void setVals(int ws[], float data[]) {
    	for(int i=0; i<3; i++)
    		this.sz[i]=ws[i];
    	this.data = new float[ws[0]*ws[1]*ws[2]*mict*moct];
    	for(int mi=0; mi<mict; mi++) {
    		for(int x=0; x<ws[0]; x++) {
    			for(int y=0; y<ws[1]; y++) {
    				for(int z=0; z<ws[2]; z++) {
    					for(int mo=0; mo<moct; mo++) {
    						this.data[x+y*ws[0]+z*ws[0]*ws[1]+mi*ws[0]*ws[1]*ws[2]+mo*ws[0]*ws[1]*ws[2]*mict] = 
    								data[mi + x*mict + y*ws[0]*mict + z*ws[0]*ws[1]*mict + mo*ws[0]*ws[1]*ws[2]*mict ];
    					}
    				}
    			}
    		}
    	}
    }
    
    public void reInitialize() {
    	Random generator = new Random();
    	int totSz = mict*moct*sz[0]*sz[1]*sz[2];
    	for(int idx=0; idx<totSz; idx++)
    		this.data[idx] = this.sigma*((float)(generator.nextGaussian()));
    }
    
    public void changeSize(int tmict, int txsz, int tysz, int tzsz, int tmoct) {
    	// This doesn't center
    	tmict = (tmict==-1)? mict : tmict;
    	txsz = (txsz==-1)? sz[0] : txsz;
    	tysz = (tysz==-1)? sz[1] : tysz;
    	tzsz = (tzsz==-1)? sz[2] : tzsz;
    	tmoct = (tmoct==-1)? moct : tmoct;    	
    	
    	int osz[] = new int[3];
    	System.arraycopy(sz,0,osz,0,3);
    	
    	sz[0] = txsz;
    	sz[1] = tysz;
    	sz[2] = tzsz;
    	
    	int omict=mict;
    	int omoct=moct;
    	
    	float odata[] = new float[omict*osz[0]*osz[1]*osz[2]*omoct];
    	System.arraycopy(data,0,odata,0,omict*osz[0]*osz[1]*osz[2]*omoct);
    	
    	int ooff[] = new int[3];
    	int noff[] = new int[3];
    	for(int i=0; i<3; i++) {
    		if(osz[i]<sz[i]) {
    			ooff[i] = 0;
    			noff[i] = (int)((sz[i]-osz[i])/2);
    		} else {
    			ooff[i] = (int)((osz[i]-sz[i])/2);
    			noff[i] = 0;
    		}
    	}
    	
    	data = new float[tmict*sz[0]*sz[1]*sz[2]*tmoct];
    	
    	
    	mict = tmict;
    	moct = tmoct;
    	this.reInitialize();
    	
    	// Reinitialize first, then copy old values?
    	for(int mi=0; mi<Math.min(mict, omict); mi++) {
    		for(int x=0; x<Math.min(sz[0],osz[0]); x++) {
    			for(int y=0; y<Math.min(sz[1],osz[1]); y++) {
    				for(int z=0; z<Math.min(sz[2],osz[2]); z++) {
    					for(int mo=0; mo<Math.min(moct, omoct); mo++) {
    						data[(x+noff[0]) + (y+noff[1])*sz[0] + (z+noff[2])*sz[0]*sz[1] + (mi)*sz[0]*sz[1]*sz[2] + (mo)*sz[0]*sz[1]*sz[2]*mict] = odata[(x+ooff[0]) + (y+ooff[1])*osz[0] + (z+ooff[2])*osz[0]*osz[1] + (mi)*osz[0]*osz[1]*osz[2] + (mo)*osz[0]*osz[1]*osz[2]*omict];
    					}
    				}
    			}
    		}
    	}
    	
    }

    private void selfAdjust() {
		cx1 = x1 + (int)((0.16)*(x2-x1)+centerPushx);
		cx2 = x2 - (int)((0.16)*(x2-x1)-centerPushx);
		cy1 = y1 - (int)((0.2)*this.eucdist(x1,y1,x2,y2)-centerPushy);
		cy2 = y2 + (int)((0.2)*this.eucdist(x1,y1,x2,y2)+centerPushy);
		xc = (int)((0.125)*x1 + (0.375)*cx1 + (0.375)*cx2 + (0.125)*x2);
        yc = (int)((0.125)*y1 + (0.375)*cy1 + (0.375)*cy2 + (0.125)*y2);
	}
	
    public void setStart(int xs, int ys) {
		x1 = xs;
		y1 = ys;
		this.selfAdjust();
    }
	
    public void setEnd(int xe, int ye) {
		x2 = xe;
		y2 = ye;
		this.selfAdjust();
    }
    
    public void setCenter(int txc, int tyc) {
    	this.centerPushx = (float) (((float)txc-0.5*(float)(x1+x2)));
    	this.centerPushy = (float) (((float)tyc-0.5*(float)(y1+y2)));
    	this.selfAdjust();
    }
    
    public void tempMove(int txc, int tyc) {
		this.x1 = txc;
		this.y1 = tyc+20;
		this.x2 = txc;
		this.y2 = tyc-20;
		this.selfAdjust();
    }
    
    public void setDest(int to, int moct) {
		this.to = to;
		this.moct = moct;
		if(this.data==null)
			this.data = new float[this.mict*this.moct*this.sz[0]*this.sz[1]*this.sz[2]];
    }
    
    public void setOrigin(int fromId) {
		from = fromId;
    }
    
    public int containsPoint(int tx, int ty) {
    	Shape objHandle;
    	objHandle = new Rectangle2D.Float(xc-7,yc-7,16,16);
		if(objHandle.contains(tx,ty))
			return 1;
    	objHandle = new Ellipse2D.Float(x1-4,y1-4,8,8);
    	if(objHandle.contains(tx,ty))
			return 2;
		objHandle = new Ellipse2D.Float(x2-4,y2-4,8,8);
		if(objHandle.contains(tx,ty))
			return 3;
    	
    	return 0;
    }

    public void selfDraw(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
			    			RenderingHints.VALUE_ANTIALIAS_ON);
		if(selected==false) {
	    	g2.setColor(Color.black);
		} else {
	    	g2.setColor(new Color(200,200,0));
		}
		CubicCurve2D tempConn = new CubicCurve2D.Float(x1,y1,cx1,cy1,cx2,cy2,x2,y2);
		g2.draw(tempConn);
		Rectangle2D connHandle = new Rectangle2D.Float(xc-7,yc-8,16,16);
		g2.setColor(Color.lightGray);
		g2.fill(connHandle);
		if(selected==false) {
	    	g2.setColor(Color.black);
		} else {
	    	g2.setColor(Color.yellow);
		}
		g2.draw(connHandle);
		g2.setColor(Color.black);
		g.setFont(new Font("Verdana", Font.PLAIN, 12));
		g.drawString("W",xc-4,yc+5);
		if(to==-1) {
			Ellipse2D endHandle = new Ellipse2D.Float(x2-4,y2-4,8,8);
			g2.setColor(Color.red);
			g2.draw(endHandle);
			if(moct != null) {
				g.setFont(new Font("Verdana", Font.PLAIN, 9));
				g.drawString(Integer.toString(moct),x2+6,y2+3);
			}
		}
		if(from==-1) {
			Ellipse2D startHandle = new Ellipse2D.Float(x1-4,y1-4,8,8);
			g2.setColor(Color.blue);
			g2.draw(startHandle);
			if(mict != null) {
				g.setFont(new Font("Verdana", Font.PLAIN, 9));
				g.drawString(Integer.toString(mict),x1+6,y1+3);
			}
		}
    }

    private float eucdist(int x1, int y1, int x2, int y2) {
        return (float)(Math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)));
    }
}