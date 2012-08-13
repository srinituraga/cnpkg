import java.awt.Graphics;


public abstract class DAGelement {
	public float eta;
	public float sigma;
	public boolean active;
    public boolean selected;
    public String name;
    public Integer sz[];
    public int id;
    
    protected int xc, yc;
    
	public DAGelement(String name, float sigma, float eta, int id, boolean active, boolean selected) {
		this.name = name;
		this.sigma = sigma;
		this.eta = eta;
		this.id = id;
		this.active = active;
		this.selected = selected;
	}
	
	abstract void selfDraw(Graphics g);
}
