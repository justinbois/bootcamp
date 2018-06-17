import numpy as np
import pandas as pd
import altair as alt

def _outliers(data):
    bottom, middle, top = np.percentile(data, [25, 50, 75])
    iqr = top - bottom
    top_whisker = min(top + 1.5*iqr, data.max())
    bottom_whisker = max(bottom - 1.5*iqr, data.min())
    outliers = data[(data > top_whisker) | (data < bottom_whisker)]
    return outliers

def _box_and_whisker(data):
    middle = data.median()
    bottom = data.quantile(0.25)    
    top = data.quantile(0.75)
    iqr = top - bottom
    top_whisker = min(top + 1.5*iqr, data.max())
    bottom_whisker = max(bottom - 1.5*iqr, data.min())
    return pd.Series({'middle': middle, 
                      'bottom': bottom, 
                      'top': top, 
                      'top_whisker': top_whisker, 
                      'bottom_whisker': bottom_whisker})

def _jitter(x, jitter_width=0.2):
    """Make x-coordinates for a jitter plot."""
    return (pd.Categorical(x).codes 
            + np.random.uniform(low=-jitter_width, high=jitter_width, size=len(x)))


def _ecdf_vals(data, formal=False):
    """
    Get x, y, values of an ECDF for plotting.

    Parameters
    ----------
    data : ndarray
        One dimensional Numpay array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)

    if formal:
        # Set up output arrays
        x_formal = np.empty(2*(len(x) + 1))
        y_formal = np.empty(2*(len(x) + 1))

        # y-values for steps
        y_formal[:2] = 0
        y_formal[2::2] = y
        y_formal[3::2] = y

        # x- values for steps
        x_formal[0] = x[0]
        x_formal[1] = x[0]
        x_formal[2::2] = x
        x_formal[3:-1:2] = x[1:]
        x_formal[-1] = x[-1]

        return x_formal, y_formal
    else:
        return x, y


def altair_ecdf():
    pass


def altair_jitter(data=None, encode_x=None, encode_y=None, 
                  encode_tooltip=alt.Tooltip(),
                  height=alt.utils.schemapi.Undefined,
                  width=alt.utils.schemapi.Undefined, jitter_width=0.2):
    """Generate a jitter plot with Altair.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy data frame.
    encode_x : str or altair.X instance
        Vega-Lite specification of x-values.
    encode_y : str or altair.Y instance
        Vega-Lite specification of y-values.
    encode_tooltip : list or altair.Tooltip instance
        Specification for tooltips.
    height : float or Undefined (default)
        Height of the chart, in pixels.
    width : float or Undefined (default)
        Width of the chart, in pixels.
    jitter_width : float
        Maximum jitter distance; must be between 0 and 0.5 to avoid 
        clashes.

    Returns
    -------
    output : Chart
        Altair Chart instance.
    """
    if data is None:
        raise RuntimeError('`data` must be specified.')
    if encode_x is None:
        raise RuntimeError('`encode_x` must be specified.')
    if encode_y is None:
        raise RuntimeError('`encode_y` must be specified.')
    if not (0 <= jitter_width <= 0.5):
        raise RuntimeError('Must have `jitter_width` between 0 and 0.5.')

    # Make Altair instances
    if isinstance(encode_x, alt.X):
        x = encode_x
    else:
        x = alt.X(encode_x)

    if isinstance(encode_y, alt.Y):
        y = encode_y
    else:
        y = alt.Y(encode_y)

    # Get column names
    if len(x.shorthand) > 1 and x.shorthand[-2] == ':':
        x_name = x.shorthand[:-2]
    else:
        x_name = x.shorthand

    if len(y.shorthand) > 1 and y.shorthand[-2] == ':':
        y_name = y.shorthand[:-2]
    else:
        y_name = y.shorthand

    # Determine types
    var_types = [None, None]
    for i, var in enumerate([x, y]):
        if not isinstance(var.type, alt.utils.schemapi.UndefinedType):
            var_types[i] = var.type[0].upper()
        elif len(var.shorthand) > 1 and var.shorthand[-2] == ':':
            var_types[i] = var.shorthand[-1]
        else:
            raise RuntimeError(
                    f'Data type of `encode_{var}` must be specified.')

    # Make sure data types are given and ok
    if var_types[0] not in 'NO' and var_types[1] not in 'NO':
        raise RuntimeError('Either `x` or `y` must be nominal or ordinal.')
    if var_types == ['N, N']:
        raise RuntimeError('Cannot have both `x` and `y` be nominal.')

    # Decide if it's a horizontal plot or not
    if var_types[0] in 'NO':
        horizontal = False
        cats = x_name
        val = y_name
        if isinstance(y.title, alt.utils.schemapi.UndefinedType):
            y.title = y_name
    else:
        horizontal = True
        cats = y_name
        val = x_name
        if isinstance(x.title, alt.utils.schemapi.UndefinedType):
            x.title = x_name

    # Copy DataFrame so we don't overwrite anything
    df = data.copy()

    # Set up groupby object
    n_cats = len(df[cats].unique())
    nominal_axis_vals = list(range(n_cats))

    # Make coordinates for plotting
    df['__jitter'] = _jitter(df[cats], jitter_width)


    if horizontal:
        chart = alt.Chart(
                data=df,
                width=width,
                height=height              
            ).mark_point(
            ).encode(
                y=alt.Y(
                    '__jitter:Q',
                    axis=alt.Axis(
                        title=None,
                        values=nominal_axis_vals,
                        labels=False,
                        grid=False,
                        ticks=False)),
                x=x,
                color=alt.Color(f'{cats}:N', title=y.title),
                tooltip=encode_tooltip)
    else:
        chart = alt.Chart(
                data=df,
                width=width,
                height=height              
            ).mark_point(
            ).encode(
               x=alt.X(
                    '__jitter:Q',
                    axis=alt.Axis(
                        title=None,
                        values=nominal_axis_vals,
                        labels=False,
                        grid=False,
                        ticks=False)),
                y=y,
                color=alt.Color(f'{cats}:N', title=x.title),
                tooltip=encode_tooltip)
    return chart


def altair_box(data=None, encode_x=None, encode_y=None, 
               encode_color=alt.Color(), height=None, width=None):
    """Generate a box plot with Altair.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy data frame.
    encode_x : str or altair.X instance
        Specification of x-values.
    encode_y : str or altair.Y instance
        Specification of y-values.
    encode_color : str or Color instance or None or Undefined (default)
        Specification of coloring of box plot. If Undefined (Default),
        all boxes are colored with Altair defaults. If None, the boxes
        are colored according to the categorical variable.
    height : float or None (default)
        Height of the chart, in pixels. If None, inferred.
    width : float or None (default)
        Width of the chart, in pixels. If None, inferred.

    Returns
    -------
    output : Chart
        Altair Chart instance.
    """

    # Make Altair instances
    if isinstance(encode_x, alt.X):
        x = encode_x
    else:
        x = alt.X(encode_x)

    if isinstance(encode_y, alt.Y):
        y = encode_y
    else:
        y = alt.Y(encode_y)

    # Get column names
    if len(x.shorthand) > 1 and x.shorthand[-2] == ':':
        x_name = x.shorthand[:-2]
    else:
        x_name = x.shorthand

    if len(y.shorthand) > 1 and y.shorthand[-2] == ':':
        y_name = y.shorthand[:-2]
    else:
        y_name = y.shorthand

    # Get axis titles
    if isinstance(x.title, alt.utils.schemapi.UndefinedType):
        x_title = x_name
    else:
        x_title = x.title
    if isinstance(y.title, alt.utils.schemapi.UndefinedType):
        y_title = y_name
    else:
        y_title = y.title

    # Determine types
    var_types = [None, None]
    for i, var in enumerate([x, y]):
        if not isinstance(var.type, alt.utils.schemapi.UndefinedType):
            var_types[i] = var.type[0].upper()
        elif len(var.shorthand) > 1 and var.shorthand[-2] == ':':
            var_types[i] = var.shorthand[-1]
        else:
            raise RuntimeError(
                    f'Data type of `encode_{var}` must be specified.')

    # Make sure data types are given and ok
    if var_types[0] not in 'NO' and var_types[1] not in 'NO':
        raise RuntimeError('Either `x` or `y` must be nominal or ordinal.')
    if var_types == ['N, N']:
        raise RuntimeError('Cannot have both `x` and `y` be nominal.')

    # Decide if it's a horizontal plot or not
    if var_types[0] in 'NO':
        horizontal = False
        cats = x_name
        val = y_name
        if encode_color is None:
            encode_color = alt.Color(f'{cats}:N', title=x.title)
    else:
        horizontal = True
        cats = y_name
        val = x_name
        if encode_color is None:
            encode_color = alt.Color(f'{cats}:N', title=y.title)

    # Set up groupby object
    grouped = data.groupby(cats)
    n_boxes = len(grouped)

    # Set default heights and widths, also of bars
    if width is None:
        if horizontal:
            width = 400
        else:
            width = 200
    if height is None:
        if horizontal:
            height = 200
        else:
            height = 300

    if horizontal:
        size = height*0.9 / n_boxes
    else:
        size = width*0.9 / n_boxes
            
    # Data frame for boxes and whiskers
    df_box = (grouped[val].apply(_box_and_whisker)
                          .reset_index()
                          .rename(columns={'level_1': 'box_val'})
                          .pivot(index=cats, columns='box_val'))
    df_box.columns = df_box.columns.get_level_values(1)
    df_box = df_box.reset_index()

    # Data frame for outliers
    df_outlier = grouped[val].apply(_outliers).reset_index(level=0)

    if horizontal:
        chart_box = alt.Chart(
                data=df_box,
                width=width,
                height=height              
            ).mark_bar(
                size=size
            ).encode(
                y=alt.Y(f'{cats}:N', title=y_title),
                x=alt.X('bottom:Q', title=x_title),
                x2=alt.X2('top:Q', title=x_title),
                color=encode_color)

        chart_median = alt.Chart(
                data=df_box,
                width=width,
                height=height              
            ).mark_tick(
                size=size,
                color='white'
            ).encode(
                y=alt.Y(f'{cats}:N',  title=y_title),
                x=alt.X('middle:Q', title=x_title))

        chart_whisker = alt.Chart(
                data=df_box,
                width=width,
                height=height              
            ).mark_rule(
            ).encode(
                y=alt.Y(f'{cats}:N', title=y_title),
                x=alt.X('bottom_whisker:Q', title=x_title),
                x2=alt.X2('top_whisker:Q', title=x_title))

        chart_outliers = alt.Chart(
                data=df_outlier,
                width=width,
                height=height              
            ).mark_point(
            ).encode(
                y=alt.Y(f'{cats}:N', title=y_title),
                x=alt.X(f'{val}:Q', title=x_title),
                color=encode_color)
    else:
        chart_box = alt.Chart(
                data=df_box,
                width=width,
                height=height       
            ).mark_bar(
                size=size
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y('bottom:Q', title=y_title),
                y2=alt.Y2('top:Q', title=y_title),
                color=encode_color)

        chart_median = alt.Chart(
                data=df_box,
                width=width,
                height=height       
            ).mark_tick(
                size=size,
                color='white'
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y('middle:Q', title=y_title))

        chart_whisker = alt.Chart(
                data=df_box,
                width=width,
                height=height       
            ).mark_rule(
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y('bottom_whisker:Q', title=y_title),
                y2=alt.Y2('top_whisker:Q', title=y_title))

        chart_outliers = alt.Chart(
                data=df_outlier,
                width=width,
                height=height       
            ).mark_point(
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y(f'{val}:Q', title=y_title),
                color=encode_color)

    return chart_whisker + chart_box + chart_median + chart_outliers