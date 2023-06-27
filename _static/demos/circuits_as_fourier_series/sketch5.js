///// Sketch 11

var sketch11 = function(p) {
    var mouse1 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg11_0 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier11-0.png');
	bg11_1 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier11-1.png');
	
    }

    p.draw = function() {
	str1 = mouse1.toString();
	picName = "bg11_" + str1;
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 200) && (p.mouseX < 250) && (p.mouseY > 140) && (p.mouseY < 190)) {
	    mouse1 = (mouse1 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch11, 'sketch11');

///// Sketch 11_5

var sketch11_5 = function(p) {
    var mouse1 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg11_2 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier11-2.png');
	bg11_3 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier11-3.png');
	
    }

    p.draw = function() {
	str1 = (mouse1+2).toString();
	picName = "bg11_" + str1;
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
	    mouse1 = (mouse1 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch11_5, 'sketch11_5');

///// Sketch 12

var sketch12 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var theta = 0;
    var rad = 57;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg12 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier12.png');
	label = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier12-diff.png');
    }

    p.draw = function() {
	p.background(bg12);

	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    theta = 2*3.15*p.mouseX/width;
	} 
	horz = p.cos(theta);
	vert = p.sin(theta);

	p.stroke(255, 181, 241);
	p.strokeWeight(2);
	p.line(222 + rad*horz, 153 - rad*vert, 222, 153);
	p.line(222 + rad*horz, 153 - rad*vert, 222, 153 - 2*rad*vert);
	p.stroke(199, 86, 178);
	p.line(222, 153, 222, 155 - 2*rad*vert);

	p.noStroke();
	p.fill(215, 162, 246);
	p.circle(p.mouseX, 255, 10);
	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(222, 153, 30, 30, -theta, 0);

	p.image(label, 222 - 20, 153 - 2*rad*vert - 40, label.width/3.5, label.height/3.5);
    }
}

var myp5 = new p5(sketch12, 'sketch12');
