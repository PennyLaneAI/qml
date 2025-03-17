r"""Density Matrix Embedding Theory (DMET)
=========================================
Materials simulation presents a crucial challenge in quantum chemistry, as understanding and predicting the properties of 
complex materials is essential for advancements in technology and science. While Density Functional Theory (DFT) is 
the current workhorse in this field due to its balance between accuracy and computational efficiency, it often falls short in 
accurately capturing the intricate electron correlation effects found in strongly correlated materials. As a result, 
researchers often turn to more sophisticated methods, such as full configuration interaction or coupled cluster theory, 
which provide better accuracy but come at a significantly higher computational cost. Embedding theories provide a balanced 
midpoint solution that enhances our ability to simulate materials accurately and efficiently. The core idea behind embedding 
is to treat the strongly correlated subsystem accurately using high-level quantum mechanical methods while approximating
the effects of the surrounding environment in a way that retains computational efficiency. 
Density matrix embedding theory(DMET) is one such efficient wave-function-based embedding approach to treat strongly 
correlated systems. Here, we present a demonstration of how to run DMET calculations through an existing library called 
libDMET, along with the intructions on how we can use the generated Hamiltonian with PennyLane to use it with quantum 
computing algorithms. We begin by providing a high-level introduction to DMET, followed by a tutorial on how to set up 
a DMET calculation."""