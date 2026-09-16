from eso.melate.sources.melate_e import parse_prize_page


HTML = r'''
<html><body>
<h1>Resultado del sorteo número 4265 de Melate</h1>
<p>La combinación ganadora del sorteo 4265 de Melate es: 20, 21, 27, 28, 37, 42. Con el número adicional: 16</p>
<h2>Fecha del sorteo: domingo 13 de septiembre de 2026</h2>
<table>
<thead><tr><th>Lugar</th><th>Aciertos</th><th>Número de ganadores</th><th>Premio individual</th></tr></thead>
<tbody>
<tr><td>1er</td><td>6 números naturales</td><td>0</td><td>$0.00</td></tr>
<tr><td>2ndo</td><td>5 números naturales + adicional</td><td>0</td><td>$0.00</td></tr>
<tr><td>3ro</td><td>5 números naturales</td><td>8</td><td>$83,447.27</td></tr>
<tr><td>4to</td><td>4 números naturales + adicional</td><td>18</td><td>$4,279.34</td></tr>
<tr><td>5to</td><td>4 números naturales</td><td>381</td><td>$1,280.43</td></tr>
<tr><td>6to</td><td>3 números naturales y el adicional</td><td>739</td><td>$161.29</td></tr>
<tr><td>7mo</td><td>3 números naturales</td><td>10,739</td><td>$43.01</td></tr>
<tr><td>8vo</td><td>2 números naturales y el adicional</td><td>9,603</td><td>$32.26</td></tr>
<tr><td>9no</td><td>2 números naturales</td><td>102,688</td><td>$26.88</td></tr>
</tbody></table>
</body></html>
'''


def test_parse_prize_page():
    meta, tiers = parse_prize_page(HTML, "fixture://4265", expected_contest=4265)
    assert meta["main"] == (20, 21, 27, 28, 37, 42)
    assert meta["additional"] == 16
    assert len(tiers) == 9
    assert int(tiers.loc[tiers.tier == 7, "winners"].iloc[0]) == 10739
    assert float(tiers.loc[tiers.tier == 3, "prize_individual"].iloc[0]) == 83447.27
