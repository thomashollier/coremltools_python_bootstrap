import numpy as np
import coremltools as ct
import PIL.Image
import sys 

COMPILE_MODELS=1

if COMPILE_MODELS:
	import shutil
	for m in ["models/DeepLabV3.mlmodel", "models/DepthAnythingV2SmallF16.mlpackage", "models/DETRResnet50SemanticSegmentationF16.mlpackage"]:
		print("\n--  Compiling model: %s" % m)
		model = ct.models.MLModel(m)
		path = model.get_compiled_model_path()
		dest = m.replace("models", "modelsC").replace("mlmodel", "mlmodelc").replace("mlpackage","mlmodelc")
		print("path: %s" % path)
		print("dest: %s" % dest)
		shutil.copytree(path, dest, dirs_exist_ok=True)
		print("Initializing compiled model")
		model = ct.models.CompiledMLModel(dest)


depthAnything = 1
semSeg = 1
deepLab = 1



if depthAnything:
	mod = "modelsC/DepthAnythingV2SmallF16.mlmodelc"
	print("\nRunning: %s" % mod)
	model = ct.models.CompiledMLModel(mod)

	for n in range(300):
		sys.stdout.write("\rframe %s " % n)
		sys.stdout.flush()
		img = PIL.Image.open("pix/test.%05d.jpg" % int(n+1) )
		img = img.resize((518, 392))
		out = model.predict({'image':img})
		x = np.asarray(out['depth'])
		x = (x*65536).astype(np.uint16)
		theDepth = PIL.Image.fromarray(x)
		theDepth.save("pix/depthiAnything.%05d.png" % n)

if semSeg:
	mod = "modelsC/DETRResnet50SemanticSegmentationF16.mlmodelc"
	print("\nRunning: %s" % mod)
	model = ct.models.CompiledMLModel(mod)

	for n in range(300):
		sys.stdout.write("\rframe %s " % n)
		sys.stdout.flush()
		img = PIL.Image.open("pix/test.%05d.jpg" % int(n+1) )
		img = img.resize((448,448))
		out = model.predict({'image':img})
		tmp = 255*out['semanticPredictions'].astype(np.float32)/200 
		tmp = tmp.astype(np.uint8)
		thePic = PIL.Image.fromarray(tmp)
		thePic.save("pix/DETR_semSeg.%05d.png" % int(n+1) )

if deepLab:
	mod = "modelsC/DeepLabV3.mlmodelc"
	print("\nRunning: %s" % mod)
	model = ct.models.CompiledMLModel(mod)
	for n in range(300):
		sys.stdout.write("\rframe %s " % n)
		sys.stdout.flush()
		img = PIL.Image.open("pix/test.%05d.jpg" % int(n+1) )
		img = img.resize((513,513))
		out = model.predict({'image':img})
		tmp = (out['semanticPredictions']*255/22).astype(np.uint8) 
		thePic = PIL.Image.fromarray(tmp)
		thePic.save("pix/deepLab.%05d.png" % int(n+1) )

