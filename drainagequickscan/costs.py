import pandas as pd
import numpy as np
from .util import rounder


def create_cost_csv(io, sheetname, csvname):
    cnum = "0" + sheetname[0]
    df = pd.read_excel(io, sheet_name=sheetname, skiprows=23, header=None, usecols=range(1, 19))
    columns = ['code post', 'omschrijving post', 'freq.', 'type', 'vanaf', 't/m', 'looptijd', 'check', 'hoeveelheid',
               'eenheid', 'prijs', 'factor', 'aantal', 'lengte', 'hoogte', 'breedte', 'opp', 'totaal']
    df.rename(columns={i: v for i, v in enumerate(columns)}, inplace=True)

    df.dropna(axis=0, how="any", subset=["code post"], inplace=True)
    df["code post"] = df["code post"].apply(str)
    df.set_index("code post", inplace=True)

    end_ind = np.where(df.index == "INV{}".format(cnum))
    df.drop(df.index[np.amax(end_ind)+1:], inplace=True)
    for icol in df.loc[:, "factor":"opp"].columns:
        df[icol] = pd.to_numeric(df[icol], errors="coerce")

    df.to_csv(csvname)
    return


def costs_vfilt(y_hoh_vp, length, nvp, nvp_per_inspectieput, L_filter, picklename=None, difficulty="normaal"):
    """
    :param io: path to excelfile
    :param y_hoh_vp: distance between vertical wells
    :param length: total length of system parallel to dike
    :param nvp: number of vertical wells
    :param nvp_per_inspectieput: number of inspection wells divided by number vertical wells
    :param L_filter: length filter
    :return:

    """
    if picklename is None:
        picklename = r"./costtable_vertfilters.pkl"
    df = pd.read_pickle(picklename)

    # Lengte totale systeem invullen:
    df.loc["200110", "lengte"] = length
    df.loc["300110", "lengte"] = length
    df.loc["500140", "lengte"] = length
    df.loc["500160", "lengte"] = length

    # aantal putten invullen
    df.loc["200120", "aantal"] = nvp
    df.loc["200130", "aantal"] = nvp
    df.loc["200140", "aantal"] = nvp
    df.loc["300120", "aantal"] = nvp
    df.loc["300130", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["300140", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["400110", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["400130", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["500110", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["500130", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["500180", "aantal"] = nvp/nvp_per_inspectieput
    df.loc["500190", "aantal"] = nvp

    # factoren invullen
    df.loc["200110", "factor"] = 1. / y_hoh_vp
    df.loc["300120", "factor"] = 1./nvp_per_inspectieput

    # lengte filter invoeren
    df.loc["200110", "hoogte"] = L_filter

    totlen = length + df.loc["400110", "factor":"opp"].product(axis=0, skipna=True) + df.loc["200110", "factor":"opp"].product(axis=0, skipna=True)

    df.loc["600110", "lengte"] = totlen
    df.loc["200220", "lengte"] = totlen

    # Update hoeveelheden
    df.loc[:"80", "hoeveelheid"] = df.loc[:"80", "factor":"opp"].product(axis=1, skipna=True)
    df.loc["500190", "hoeveelheid"] = np.round(df.loc["500190", "hoeveelheid"])

    # Update totals
    df.loc[:"80", "totaal"] = df.loc[:"80", "hoeveelheid"] * df.loc[:"80", "prijs"]

    df = extra_other_costs(df, "01", difficulty=difficulty)

    return df


def costs_boring(length, factor_inspectionwell, nbor, picklename=None, difficulty="normaal"):
    """
    :param io: path to excelfile
    :param y_hoh_vp: distance between vertical wells
    :param length: total length of system parallel to dike
    :param nvp: number of vertical wells
    :param nvp_per_inspectieput: number of inspection wells divided by number vertical wells
    :param L_filter: length filter
    :return:

    """
    if picklename is None:
        picklename = r"./costtable_boring.pkl"
    df = pd.read_pickle(picklename)

    # Lengte totale systeem invullen:
    df.loc["200250", "lengte"] = length*nbor
    df.loc["300220", "lengte"] = length
    df.loc["500240", "lengte"] = length
    df.loc["500260", "lengte"] = length

    # aantal putten invullen
    df.loc["300230", "aantal"] = length*factor_inspectionwell
    df.loc["300240", "aantal"] = length*factor_inspectionwell
    df.loc["400210", "aantal"] = length*factor_inspectionwell
    df.loc["400230", "aantal"] = length*factor_inspectionwell
    df.loc["500210", "aantal"] = length*factor_inspectionwell
    df.loc["500230", "aantal"] = length*factor_inspectionwell
    df.loc["500270", "aantal"] = length*factor_inspectionwell

    # factoren invullen
    df.loc["300220", "factor"] = factor_inspectionwell

    totlen = length + df.loc["400210", "factor":"opp"].product(axis=0, skipna=True)

    df.loc["600210", "lengte"] = 1.2 * length
    df.loc["200320", "lengte"] = totlen*nbor

    # Update hoeveelheden
    df.loc[:"80", "hoeveelheid"] = df.loc[:"80", "factor":"opp"].product(axis=1, skipna=True)

    # Update totals
    df.loc[:"80", "totaal"] = df.loc[:"80", "hoeveelheid"] * df.loc[:"80", "prijs"]

    df = extra_other_costs(df, "02", difficulty=difficulty)

    return df


def costs_dig(length, factor_inspectionwell, nsl, picklename=None, difficulty="normaal"):
    """

     :param io: path to excelfile
     :param y_hoh_vp: distance between vertical wells
     :param length: total length of system parallel to dike
     :param nvp: number of vertical wells
     :param nvp_per_inspectieput: number of inspection wells divided by number vertical wells
     :param L_filter: length filter
     :return:

     """
    if picklename is None:
        picklename = r"./costtable_dig.pkl"
    df = pd.read_pickle(picklename)

    # Lengte totale systeem invullen:
    df.loc["200360", "lengte"] = length * nsl
    df.loc["300320", "lengte"] = length

    totlen = length + df.loc["400310", "factor":"opp"].product(axis=0, skipna=True)

    df.loc["600310", "lengte"] = totlen
    df.loc["200420", "lengte"] = totlen * nsl

    # aantal putten invullen
    df.loc["300330", "aantal"] = length*factor_inspectionwell
    df.loc["300340", "aantal"] = length*factor_inspectionwell
    df.loc["400310", "aantal"] = length*factor_inspectionwell
    df.loc["400330", "aantal"] = length*factor_inspectionwell
    df.loc["500310", "aantal"] = length*factor_inspectionwell
    df.loc["500330", "aantal"] = length*factor_inspectionwell
    df.loc["500370", "aantal"] = length*factor_inspectionwell
    df.loc["500380", "aantal"] = length*factor_inspectionwell

    # factoren invullen
    df.loc["300320", "factor"] = factor_inspectionwell
    df.loc["500360", "factor"] = factor_inspectionwell

    # Update hoeveelheden
    df.loc[:"80", "hoeveelheid"] = df.loc[:"80", "factor":"opp"].product(axis=1, skipna=True)

    # Update totals
    df.loc[:"80", "totaal"] = df.loc[:"80", "hoeveelheid"] * df.loc[:"80", "prijs"]

    df = extra_other_costs(df, "03", difficulty=difficulty)

    return df


def costs_grindkoffer(length, z_deklaag_boven, z_deklaag_onder, x_breedte_gk,
                      talud_gk, factor_stuw, picklename=None, difficulty="normaal"):
    """

     :param io: path to excelfile
     :param y_hoh_vp: distance between vertical wells
     :param length: total length of system parallel to dike
     :param nvp: number of vertical wells
     :param nvp_per_inspectieput: number of inspection wells divided by number vertical wells
     :param L_filter: length filter
     :return:

     """
    if picklename is None:
        picklename = r"./costtable_gk.pkl"
    df = pd.read_pickle(picklename)

    hoogte_gk = z_deklaag_boven - z_deklaag_onder
    x_breedte_gk_boven = hoogte_gk / talud_gk * 2 + x_breedte_gk
    A_gk = (x_breedte_gk + x_breedte_gk_boven) / 2. * hoogte_gk
    # A_gk = ((1. + (x_breedte_gk/(1./talud_gk)))/x_breedte_gk) * (1. / talud_gk * hoogte_gk * x_breedte_gk) # for some reason?!
    f = A_gk / (1. / talud_gk * hoogte_gk * x_breedte_gk)
    nstuw = factor_stuw * length
    vol = A_gk * length
    surfarea = 2 * np.sqrt(hoogte_gk ** 2 + (hoogte_gk / talud_gk) ** 2) + x_breedte_gk

    # Lengte totale systeem invullen:
    df.loc["200470", "lengte"] = length
    df.loc["200480", "lengte"] = length
    df.loc["200500", "lengte"] = length
    df.loc["400440", "lengte"] = length
    df.loc["500440", "lengte"] = length
    df.loc["500460", "lengte"] = length
    df.loc["600420", "lengte"] = length

    # Overige dimensies grindkoffer
    df.loc["200470", "hoogte"] = hoogte_gk
    df.loc["200470", "factor"] = 1. / talud_gk
    df.loc["200470", "breedte"] = x_breedte_gk
    df.loc["200470", "opp"] = f

    # aantal putten invullen
    df.loc["500420", "aantal"] = nstuw
    df.loc["500430", "aantal"] = nstuw
    df.loc["500470", "aantal"] = nstuw

    # factoren invullen
    df.loc["500460", "factor"] = factor_stuw

    # volume invullen
    df.loc["200490", "opp"] = vol
    df.loc["200510", "opp"] = vol
    df.loc["200500", "opp"] = surfarea

    # Update hoeveelheden
    df.loc[:"80", "hoeveelheid"] = df.loc[:"80", "factor":"opp"].product(axis=1, skipna=True)

    # Update totals
    df.loc[:"80", "totaal"] = df.loc[:"80", "hoeveelheid"] * df.loc[:"80", "prijs"]

    df = extra_other_costs(df, "04", difficulty=difficulty)

    return df


def extra_other_costs(df, cnum, difficulty=None):

    m = cnum
    if difficulty is not None:
        df = set_difficulty(difficulty, df, cnum)

    df.loc["BKbdk", "totaal"] = df.iloc[:df.index.get_loc("BKbdk") - 1, -1].dropna().sum()
    df.loc["NTD"+m+"1", "totaal"] = df.loc["NTD"+m+"1", "hoeveelheid"] * df.loc["BKbdk", "totaal"]
    df.loc["BKdk", "totaal"] = df.loc["BKbdk", "totaal"] + df.loc["NTD"+m+"1", "totaal"]
    df.loc["IK"+m+"6", "totaal"] = df.loc["IK"+m+"6", "hoeveelheid"] * df.loc["BKdk", "totaal"]
    df.loc["IK"+m+"8", "totaal"] = df.loc["IK"+m+"8", "hoeveelheid"] * df.loc["BKdk", "totaal"]
    df.loc["IK"+m+"9", "totaal"] = df.loc["IK"+m+"9", "hoeveelheid"] * df.loc["BKdk", "totaal"]
    df.loc["IK"+m+"10", "totaal"] = (df.loc["IK"+m+"6":"IK"+m+"9", "totaal"].dropna().sum() +
                                  df.loc["BKdk", "totaal"]) * df.loc["IK"+m+"10", "hoeveelheid"]
    df.loc["IK"+m+"11", "totaal"] = (df.loc["IK"+m+"6":"IK"+m+"9", "totaal"].dropna().sum() +
                                     df.loc["BKdk", "totaal"] +
                                  df.loc["IK"+m+"10", "totaal"]) * df.loc["IK"+m+"11", "hoeveelheid"]
    df.loc["IK"+m+"12", "totaal"] = (df.loc["IK"+m+"6":"IK"+m+"9", "totaal"].dropna().sum() +
                                     df.loc["BKdk", "totaal"] +
                                  df.loc["IK"+m+"10", "totaal"]) * df.loc["IK"+m+"12", "hoeveelheid"]
    df.loc["BKik", "totaal"] = df.loc["IK"+m+"6":"IK"+m+"12", "totaal"].dropna().sum()
    df.loc["VZBK", "totaal"] = df.loc["BKdk", "totaal"] + df.loc["BKik", "totaal"]

    df.loc["RBK"+m+"3", "totaal"] = df.loc["RBK"+m+"3", "hoeveelheid"] * df.loc["VZBK", "totaal"]
    df.loc["BK"+m, "totaal"] = df.loc["VZBK", "totaal"] + df.loc["RBK"+m+"3", "totaal"]

    df.loc["EK"+m+"2", "totaal"] = df.loc["EK"+m+"2", "hoeveelheid"] * df.loc["VZBK", "totaal"]
    df.loc["EK"+m+"3", "totaal"] = df.loc["EK"+m+"3", "hoeveelheid"] * df.loc["VZBK", "totaal"]
    df.loc["EK"+m+"18", "totaal"] = df.loc["EK"+m+"18", "hoeveelheid"] * (
    df.loc["EK"+m+"2", "totaal"] + df.loc["EK"+m+"3", "totaal"])
    df.loc["EK"+m, "totaal"] = df.loc["EK"+m+"2", "totaal"] + df.loc["EK"+m+"3", "totaal"] + \
                                  df.loc["EK"+m+"18", "totaal"]
    df.loc["OK"+m+"1", "totaal"] = df.loc["OK"+m+"1", "hoeveelheid"] * df.loc["VZBK", "totaal"]
    df.loc["OK"+m+"37", "totaal"] = df.loc["OK"+m+"37", "hoeveelheid"] * df.loc["OK"+m+"1", "totaal"]
    df.loc["OBK"+m, "totaal"] = df.loc["OK"+m+"1", "totaal"] + df.loc["OK"+m+"37", "totaal"]
    df.loc["INV"+m, "totaal"] = df.loc["BK"+m, "totaal"] + df.loc["EK"+m, "totaal"] + df.loc["OBK"+m, "totaal"]

    df.loc["VK"+m, "totaal"] = 0.

    return df


def set_difficulty(difficulty, df, cnum, picklename=None):
    
    if picklename is None:
        picklename = '../data/factor_difficulty.pkl'
        
    m = cnum
    factors = pd.read_pickle(picklename)
    new_index = [s.replace("01", m) for s in factors.index]
    factors["code post"] = new_index
    factors.set_index("code post", inplace=True)
    assert difficulty in factors.columns, "difficulty must be one of: ({0}, {1}, {2})".format(*factors.columns[1:])
    df.loc[factors.index, "hoeveelheid"] = factors.loc[:, difficulty]

    return df


def costcalculations(logger, xsm, L_totaal, hdrnparams, gkparams, vpparams, difficulty="normaal"):

    y_length_tot = L_totaal

    y_hoh_vp = vpparams["welldist"]
    nwells = vpparams["nwells"]
    L_filter = vpparams["L_filter"]
    Qvp = vpparams["Qvp"]

    n_hdrns = hdrnparams["n_hdrns"]
    Qhdrn = hdrnparams["Qhdrn"]

    x_gk_breedte = gkparams["breedte_gk"]
    slope_gk = gkparams["slope_gk"]
    Qgk = gkparams["Qgk"]

    homedir = "../data"

    # Cost calculations
    logger.info("-" * 15 + " COST CALCULATIONS " + "-" * 15)
    logger.info("Difficulty set to {}".format(difficulty))

    nvp_per_inspectieput = 5  # aantal verticale putten per inspectie put.

    measures_names = {1: "Verticale bronnen", 2: "Horizontale drain (geboord)",
                      3: "Horizontale drain (gegraven)", 4: "Grindkoffer"}
    index_names = [1, 2, 4]  # edit to [2, 4] when vertical wells not included in calculations.

    costs_vp = costs_vfilt(y_hoh_vp, y_length_tot, nwells, nvp_per_inspectieput, L_filter, difficulty=difficulty,
                               picklename='{}/costtable_vertfilters.pkl'.format(homedir))
    costs_hdb = costs_boring(y_length_tot, 1 / 500., n_hdrns, difficulty=difficulty,
                                 picklename='{}/costtable_boring.pkl'.format(homedir))
    costs_gk = costs_grindkoffer(y_length_tot, xsm.z_deklaag_boven, xsm.z_deklaag_onder, x_gk_breedte, slope_gk, 1 / 500.,
                            difficulty=difficulty,
                            picklename='{}/costtable_gk.pkl'.format(homedir))

    costs_tables = {1: costs_vp, 2: costs_hdb, 4: costs_gk}
    # costs_tables = {2: costs_hdb, 4: costs_gk}  # for use when vertical wells not included

    # Hdrn met kettinggraver?
    if (xsm.z_deklaag_boven - xsm.z_deklaag_onder + 1.) <= 8.:  # if drn less than 8 m below surface
        costs_hds = costs_dig(y_length_tot, 1 / 500., n_hdrns, difficulty=difficulty,
                                  picklename='{}/costtable_dig.pkl'.format(homedir))
        # index_names.insert(2, 3)  # don't include in finalcosts table on website but do save to excel.
        costs_tables.update({3: costs_hds})  # this way it will be saved to excel.

    for j in index_names:
        m = "0" + str(j)
        idf = costs_tables[j]
        # selected_rows_costs = ("BKbdk", "BKdk", "BKik", "VZBK", "RBK", "BK"+m, "VK"+m, "EK"+m, "OBK"+m, "INV"+m)
        selected_rows_costs = ("BK" + m, "VK" + m, "EK" + m, "OBK" + m, "INV" + m)
        if j == index_names[0]:
            finalcosts = idf.loc[selected_rows_costs, ("omschrijving post", "totaal")].copy()
            finalcosts.rename(columns={"totaal": measures_names[j]}, inplace=True)
        else:
            finalcosts[measures_names[j]] = idf.loc[selected_rows_costs, "totaal"].as_matrix()
            finalcosts.rename(columns={"totaal": measures_names[j]}, inplace=True)

    finalcosts = finalcosts.iloc[:-1]
    finalcosts["omschrijving post"] = finalcosts["omschrijving post"].apply(lambda s:
                                                                            s.replace("Systeem met vertikale filters",
                                                                                      ""))
    # finalcosts["omschrijving post"] = finalcosts["omschrijving post"].apply(lambda s:
    # s.replace("Systeem met horizontaal geboord filter", ""))  # used when vert filters not presented

    # Set index without numbering
    new_index = [s.replace("0{}".format(index_names[0]), "") for s in finalcosts.index]
    finalcosts["code post"] = new_index
    finalcosts.set_index("code post", inplace=True)

    # Add maintenance costs
    finalcosts.loc["OHK", "omschrijving post"] = "Onderhoud 50 jaar"
    finalcosts.loc["OHK", "Verticale bronnen"] = 0.75 * finalcosts.loc[
        "BK", "Verticale bronnen"]  # 75% van BK, zie rapp.
    finalcosts.loc["OHK", "Horizontale drain (geboord)"] = 0.15 * finalcosts.loc[
        "BK", "Horizontale drain (geboord)"]  # 15% van BK, zie rapp.
    finalcosts.loc["OHK", "Grindkoffer"] = 0.20 * finalcosts.loc["BK", "Grindkoffer"]  # 20% van BK, zie rapp.

    # Add row costs changes to water system, set to 0 to show it is not included:
    finalcosts.loc["WSK", "omschrijving post"] = "Aanpassingen watersysteem"
    finalcosts.iloc[-1, 1:] = 0.

    # Kosten Kuubs
    kosten_per_kuub_eur = 0.5 / 100.  # 0,5 eurocent per kuub
    finalcosts.loc["QK", "omschrijving post"] = "Kosten waterbezwaar (€/dag/{0:.0f}m)".format(y_length_tot)
    finalcosts.loc["QK", "Grindkoffer"] = Qgk * kosten_per_kuub_eur
    finalcosts.loc["QK", "Horizontale drain (geboord)"] = Qhdrn * kosten_per_kuub_eur
    finalcosts.loc["QK", "Verticale bronnen"] = Qvp * kosten_per_kuub_eur

    # Reorder columns
    if finalcosts.shape[1] == 5:  # includes gegraven drains
        finalcosts = finalcosts.loc[:, ("omschrijving post", "Grindkoffer", "Horizontale drain (geboord)",
                                        "Horizontale drain (gegraven)", "Verticale bronnen")]
    elif finalcosts.shape[1] == 4:  # does not include gegraven drains
        finalcosts = finalcosts.loc[:, ("omschrijving post", "Grindkoffer", "Horizontale drain (geboord)",
                                        "Verticale bronnen")]

    # Round final_costs
    finalcosts_rounded = finalcosts.copy()
    finalcosts_rounded.iloc[:, 1:] = finalcosts_rounded.iloc[:, 1:].applymap(rounder)
    finalcosts_rounded.iloc[4, 1:] = finalcosts_rounded.iloc[:4, 1:].sum(axis=0)

    return finalcosts
