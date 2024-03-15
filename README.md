chef d'oeuvre M2
<br/>Le dossier contenant le dataset n'est pas push sur git

# Updates :

## Annotation manuelle des données de fine tuning :
- launch_annotation_process() du fichier `widget_annotation.py` permet de lancer l'annotation manuelle des données de fine tuning grâce à ipywidgets.
- La fonction affiche un par un les exemples avec les zones de basse confiance que l'utilisateur doit annoter.
- Les annotations sont sauvegardées dans le dossier 'manual_annotations' avec les timelines correspondantes ainsi que le fichier contenant l'ensemble des noms des échantillons présents dans le set de fine tuning.

## Mode et mehode pour la génération du set de fine tuning :
- mode : possibilité de choisir entre 'sample' et 'dataset' pour spécifier si on veut X% du dataset ou X% des samples.
- méthode : possibilité de choisir entre 'random' et 'lowest' pour spécifier si on veut choisir les samples aléatoirement ou ceux de plus basse confiance.

# Fine tuning :
- Sur AMI : DER pretrained : 19.2 % / DER fine-tuned : 18.9 % avec 20 epochs et un learning rate de 0.0001
- Sur Msdwild : DER pretrained : / DER fine-tuned : 

