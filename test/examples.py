from src.dto.dto import Answer, EvalSample

q_1 = "What is the two step process?"
q_2 = "What is the limitation of the two step process?"

a_1 = Answer(
    answer="father is removed of its silver and converted into a stamper",
    start=2659,
    end=2719,
)
a_2 = Answer(
    answer="limited to a few hundred vinyl pressings",
    start=2963,
    end=3003,
)

eval_sample_1 = EvalSample(
    document_id=1,
    document="""Original master discs are created by lathe-cutting: a lathe is used to cut a modulated groove into a blank record. The blank records for cutting used to be cooked up, as needed, by the cutting engineer, using what Robert K. Morrison describes as a "metallic soap," containing lead litharge, ozokerite, barium sulfate, montan wax, stearin and paraffin, among other ingredients. Cut "wax" sound discs would be placed in a vacuum chamber and gold-sputtered to make them electrically conductive for use as mandrels in an electroforming bath, where pressing stamper parts were made. Later, the French company Pyral invented a ready-made blank disc having a thin nitro-cellulose lacquer coating (approximately 7 mils thickness on both sides) that was applied to an aluminum substrate. Lacquer cuts result in an immediately playable, or processable, master record. If vinyl pressings are wanted, the still-unplayed sound disc is used as a mandrel for electroforming nickel records that are used for manufacturing pressing stampers. The electroformed nickel records are mechanically separated from their respective mandrels. This is done with relative ease because no actual "plating" of the mandrel occurs in the type of electrodeposition known as electroforming, unlike with electroplating, in which the adhesion of the new phase of metal is chemical and relatively permanent. The one-molecule-thick coating of silver (that was sprayed onto the processed lacquer sound disc in order to make its surface electrically conductive) reverse-plates onto the nickel record's face. This negative impression disc (having ridges in place of grooves) is known as a nickel master, "matrix" or "father." The "father" is then used as a mandrel to electroform a positive disc known as a "mother". Many mothers can be grown on a single "father" before ridges deteriorate beyond effective use. The "mothers" are then used as mandrels for electroforming more negative discs known as "sons". Each "mother" can be used to make many "sons" before deteriorating. The "sons" are then converted into "stampers" by center-punching a spindle hole (which was lost from the lacquer sound disc during initial electroforming of the "father"), and by custom-forming the target pressing profile. This allows them to be placed in the dies of the target (make and model) record press and, by center-roughing, to facilitate the adhesion of the label, which gets stuck onto the vinyl pressing without any glue. In this way, several million vinyl discs can be produced from a single lacquer sound disc. When only a few hundred discs are required, instead of electroforming a "son" (for each side), the "father" is removed of its silver and converted into a stamper. Production by this latter method, known as the "two-step-process" (as it does not entail creation of "sons" but does involve creation of "mothers," which are used for test playing and kept as "safeties" for electroforming future "sons") is limited to a few hundred vinyl pressings. The pressing count can increase if the stamper holds out and the quality of the vinyl is high. The "sons" made during a "three-step" electroforming make better stampers since they don't require silver removal (which reduces some high fidelity because of etching erasing part of the smallest groove modulations) and also because they have a stronger metal structure than "fathers".""",
    questions=[q_1, q_2],
    answers=[a_1, a_2],
)
