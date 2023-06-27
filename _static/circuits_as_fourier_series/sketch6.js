///// Sketch 13

var sketch13 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var theta = 0;
    var rad = 43;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg13 = p.loadImage('/_static/circuits_as_fourier_series/src/fourier13.png');
    }

    p.draw = function() {
	p.background(bg13);

	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    theta = 2*2*3.155*p.mouseX/width - 0.02;
	} 
	horz = p.cos(theta);
	vert = p.sin(theta);
	horz2 = p.cos(2*theta);
	vert2 = p.sin(2*theta);
	horz3 = p.cos(3*theta);
	vert3 = p.sin(3*theta);

	p.stroke(255, 181, 241);
	p.strokeWeight(2);
	p.line(225 + rad*horz, 90 - rad*vert, 225, 90);
	p.line(225 + rad*horz, 90 - rad*vert, 225, 90 - 2*rad*vert);
	p.stroke(199, 86, 178);
	p.line(225, 90, 225, 90 - 2*rad*vert);

	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    p.noStroke();
	    p.fill(215, 162, 246);
	    p.circle(p.mouseX, 255, 10);
	}
	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(225, 90, 30, 30, -theta, 0);

	p.stroke(255, 181, 241);
	p.strokeWeight(2);
	p.line(107 + rad*horz2, 162 - rad*vert2, 107, 162);
	p.line(107 + rad*horz2, 162 - rad*vert2, 107, 162 - 2*rad*vert2);
	p.stroke(199, 86, 178);
	p.line(107, 162, 107, 162 - 2*rad*vert2);

	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(107, 162, 30, 30, -2*theta, 0);

	p.stroke(255, 181, 241);
	p.strokeWeight(2);
	p.line(331 + rad*horz3, 162 - rad*vert3, 331, 160);
	p.line(331 + rad*horz3, 162 - rad*vert3, 331, 160 - 2*rad*vert3);
	p.stroke(199, 86, 178);
	p.line(331, 162, 331, 162 - 2*rad*vert3);

	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(331, 162, 30, 30, -3*theta, 0);
    }
}

var myp5 = new p5(sketch13, 'sketch13');

///// Sketch 14

var sketch14 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var theta = 0;
    var rad = 43;
    var angle1 = 0.02, angle2 = 0.13, angle3 = 0.4, angle4 = 0.63, angle5 = 0.7, angle6 = 0.95;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg13 = p.loadImage('/_static/circuits_as_fourier_series/src/fourier13.png');
    }

    p.draw = function() {
	p.background(bg13);

	if ((p.mouseY < height) && (p.mouseY > 0)) {
	    if ((p.mouseX > 0) && (p.mouseX < angle1*width)) {
		theta = 2*2*3.155*angle1 - 0.02;
	    }
	    if ((p.mouseX > angle1*width) && (p.mouseX < angle2*width)) {
		theta = 2*2*3.155*angle2 - 0.02;
	    }
	    if ((p.mouseX > angle2*width) && (p.mouseX < angle3*width)) {
		theta = 2*2*3.155*angle3 - 0.02;
	    }
	    if ((p.mouseX > angle3*width) && (p.mouseX < angle4*width)) {
		theta = 2*2*3.155*angle4 - 0.02;
	    }
	    if ((p.mouseX > angle4*width) && (p.mouseX < angle5*width)) {
		theta = 2*2*3.155*angle5 - 0.02;
	    }
	    else if (p.mouseX > angle5*width) {
		theta = 2*2*3.155*angle6 - 0.02;
	    }
	} 
	horz = p.cos(theta);
	vert = p.sin(theta);
	horz2 = p.cos(2*theta);
	vert2 = p.sin(2*theta);
	horz3 = p.cos(3*theta);
	vert3 = p.sin(3*theta);

	p.stroke(199, 86, 178);
	p.strokeWeight(2);
	p.line(225 + rad*horz, 90 - rad*vert, 225, 90);

	p.noStroke();
	p.fill(240, 238, 239);
	p.circle(angle1*width, 255, 10);
	p.circle(angle2*width, 255, 10);
	p.circle(angle3*width, 255, 10);
	p.circle(angle4*width, 255, 10);
	p.circle(angle5*width, 255, 10);
	p.circle(angle6*width, 255, 10);
	
	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    p.noStroke();
	    p.fill(215, 162, 246);
	    p.circle(theta*width/(2*2*3.155), 255, 10);
	}
	
	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(225, 90, 30, 30, -theta, 0);

	p.stroke(199, 86, 178);
	p.strokeWeight(2);
	p.line(107 + rad*horz2, 162 - rad*vert2, 107, 162);

	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(107, 162, 30, 30, -2*theta, 0);

	p.stroke(199, 86, 178);
	p.strokeWeight(2);
	p.line(331 + rad*horz3, 162 - rad*vert3, 331, 160);

	p.noFill();
	p.stroke(215, 162, 246);
	p.arc(331, 162, 30, 30, -3*theta, 0);
    }
}

var myp5 = new p5(sketch14, 'sketch14');

///// Sketch 14_5

var sketch14_5 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var rad = 43;
    var angle1 = 0.02, angle2 = 0.13, angle3 = 0.4, angle4 = 0.63, angle5 = 0.7, angle6 = 0.95;
    var theta = 0, angle = angle1;
    var xinit = 220, yinit = 153;
    var x = xinit, y = yinit;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg14_5 = p.loadImage('/_static/circuits_as_fourier_series/src/fourier14_5.png');
    }

    p.draw = function() {
	p.background(bg14_5);

	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    theta = 2*2*3.142*p.mouseX/width;
	} 

	p.stroke(255, 181, 241)
	p.strokeWeight(2);

	x = xinit;
	y = yinit;
	for (let i = -3; i < 4; i++) {
	    p.line(x, y, x + rad*p.cos(i*(theta-angle)), y - rad*p.sin(i*(theta-angle)));
	    x = x + rad*p.cos(i*(theta-angle));
	    y = y - rad*p.sin(i*(theta-angle));
	}
	p.stroke(199, 86, 178);
	p.line(x, y, xinit, yinit);

	p.noStroke();
	p.fill(240, 238, 239);
	p.circle(angle1*width, 255, 10);
	p.circle(angle2*width, 255, 10);
	p.circle(angle3*width, 255, 10);
	p.circle(angle4*width, 255, 10);
	p.circle(angle5*width, 255, 10);
	p.circle(angle6*width, 255, 10);
	
	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    p.noStroke();
	    p.fill(215, 162, 246);
	    p.circle(theta*width/(2*2*3.155), 255, 10);
	}

	p.mouseClicked = function() {
	    if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
		if ((p.mouseY < height) && (p.mouseY > 0)) {
		    if ((p.mouseX > 0) && (p.mouseX < angle1*width)) {
			angle = 2*2*3.142*angle1;
		    }
		    if ((p.mouseX > angle1*width) && (p.mouseX < angle2*width)) {
			angle = 2*2*3.142*angle2;
		    }
		    if ((p.mouseX > angle2*width) && (p.mouseX < angle3*width)) {
			angle = 2*2*3.142*angle3;
		    }
		    if ((p.mouseX > angle3*width) && (p.mouseX < angle4*width)) {
			angle = 2*2*3.142*angle4;
		    }
		    if ((p.mouseX > angle4*width) && (p.mouseX < angle5*width)) {
			angle = 2*2*3.142*angle5;
		    }
		    else if (p.mouseX > angle5*width) {
			angle = 2*2*3.142*angle6;
		    }
		} 
	    }
	}
	    p.noStroke();
	    p.fill(215, 162, 246);
	    p.circle(angle*width/(2*2*3.155), 255, 10);
    }
}

var myp5 = new p5(sketch14_5, 'sketch14_5');

///// Sketch 15

var sketch15 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var rad = 43;
    var angle1 = 1/7, angle2 = 2/7, angle3 = 3/7, angle4 = 4/7, angle5 = 5/7, angle6 = 6/7;
    var theta = 0, angle = 2*2*3.142*angle1;
    var xinit = 220, yinit = 153;
    var x = xinit, y = yinit;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg14_5 = p.loadImage('/_static/circuits_as_fourier_series/src/fourier14_5.png');
    }

    p.draw = function() {
	p.background(bg14_5);

	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    theta = 2*2*3.14*p.mouseX/width;
	} 

	p.stroke(255, 181, 241)
	p.strokeWeight(2);

	x = xinit;
	y = yinit;
	for (let i = -3; i < 4; i++) {
	    p.line(x, y, x + rad*p.cos(i*(theta-angle)), y - rad*p.sin(i*(theta-angle)));
	    x = x + rad*p.cos(i*(theta-angle));
	    y = y - rad*p.sin(i*(theta-angle));
	}
	p.stroke(199, 86, 178);
	p.line(x, y, xinit, yinit);

	p.noStroke();
	p.fill(240, 238, 239);
	p.circle(angle1*width, 255, 10);
	p.circle(angle2*width, 255, 10);
	p.circle(angle3*width, 255, 10);
	p.circle(angle4*width, 255, 10);
	p.circle(angle5*width, 255, 10);
	p.circle(angle6*width, 255, 10);
	
	if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
	    p.noStroke();
	    p.fill(215, 162, 246);
	    p.circle(theta*width/(2*2*3.155), 255, 10);
	}

	p.mouseClicked = function() {
	    if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
		if ((p.mouseY < height) && (p.mouseY > 0)) {
		    if ((p.mouseX > 0) && (p.mouseX < angle1*width)) {
			angle = 2*2*3.142*angle1;
		    }
		    if ((p.mouseX > angle1*width) && (p.mouseX < angle2*width)) {
			angle = 2*2*3.142*angle2;
		    }
		    if ((p.mouseX > angle2*width) && (p.mouseX < angle3*width)) {
			angle = 2*2*3.142*angle3;
		    }
		    if ((p.mouseX > angle3*width) && (p.mouseX < angle4*width)) {
			angle = 2*2*3.142*angle4;
		    }
		    if ((p.mouseX > angle4*width) && (p.mouseX < angle5*width)) {
			angle = 2*2*3.142*angle5;
		    }
		    else if (p.mouseX > angle5*width) {
			angle = 2*2*3.142*angle6;
		    }
		} 
	    }
	}
	    p.noStroke();
	    p.fill(215, 162, 246);
	    p.circle(angle*width/(2*2*3.155), 255, 10);
    }
}

var myp5 = new p5(sketch15, 'sketch15');
