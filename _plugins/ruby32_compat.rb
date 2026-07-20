# Local-only Ruby 3.2+ compatibility for github-pages / liquid 4.0.3
# (taint tracking was removed; Liquid still calls String#tainted?)
class Object
  def tainted?
    false
  end

  def taint
    self
  end

  def untaint
    self
  end
end
