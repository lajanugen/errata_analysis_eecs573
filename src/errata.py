
class Error:
  # fields - dictionary of field names to field data, e.g. {'effect':'machine does not turn on', 'details':'the power block is broken'}
  def __init__(self, fields={}):
    self.fields = fields

  def get_field(self, field):
    if field in self.fields:
      return self.fields[field]
    else:
      return ''

  
